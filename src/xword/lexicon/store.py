"""The scored word list the solver fills grids from.

A :class:`Lexicon` is a set of upper-case ``A-Z`` answers, each with a quality
score, plus the :class:`~xword.lexicon.index.PatternIndex` built over them. It
is the non-LLM half of candidate generation: it never knows what a clue means,
but it knows what can physically go in a slot and which of those options a real
constructor would actually use.

Score scale
-----------
Scores live in ``[0, 1]``: ``1.0`` is fill you would expect in a Monday puzzle,
``0.5`` is an ordinary dictionary word, ``0.0`` is junk that is technically a
word. Sources disagree about scale, so everything is normalised on the way in:

* a value already in ``[0, 1]`` is taken as-is;
* a value above ``1`` is assumed to be the 0-100 convention that published
  constructor word lists use, and is divided by 100 (so ``50`` -> ``0.5``);
* a missing score becomes :data:`DEFAULT_LOADED_SCORE`, i.e. "present but
  unrated" rather than "known good";
* anything out of range after that is clamped, and ``NaN`` is dropped.

Scores are only ever compared within a slot, so the absolute numbers matter
less than the ordering -- but keeping them on one bounded scale is what lets
:meth:`Lexicon.letters_at` sum them into a per-square mass that fusion can treat
as a prior weight.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from xword import config
from xword.lexicon.index import PatternIndex

__all__ = [
    "BUILTIN_FALLBACK",
    "DEFAULT_LOADED_SCORE",
    "Lexicon",
    "LexiconEntry",
    "parse_score_line",
]

#: Score for a word that arrives without one. Deliberately mid-scale: an
#: unrated word should lose to attested crossword fill and beat nothing.
DEFAULT_LOADED_SCORE = 0.5

_FIELD_SPLIT = re.compile(r"[;,\t]")


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    """One word with its score and, when it was mined from puzzles, how often it
    was seen. ``frequency`` is kept beside the score because the score is a
    lossy, normalised view of it and the harness likes to report both."""

    word: str
    score: float
    frequency: int = 0


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _parse_score(text: str) -> float | None:
    """Normalise one score field onto ``[0, 1]``; ``None`` if it is not a number."""
    try:
        value = float(text.strip())
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    if value > 1.0:
        value = value / 100.0
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def parse_score_line(line: str) -> tuple[str, float | None] | None:
    """Split one word-list line into ``(word, score or None)``.

    Accepts ``WORD;score`` (what :meth:`Lexicon.save` writes), the comma and tab
    variants that other tools emit, and a bare word. The word field is returned
    raw -- spaces and punctuation included -- because a foreign list writes
    ``It is a deal;90`` and only :func:`xword.lexicon.build.normalise_answer`
    should decide how that becomes an answer. Blank and ``#`` lines give
    ``None``.
    """
    text = line.strip()
    if not text or text.startswith("#"):
        return None

    parts = _FIELD_SPLIT.split(text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), _parse_score(parts[1])

    # No delimiter: a trailing numeric token is a score, anything else is part
    # of a multi-word answer.
    head, sep, tail = text.rpartition(" ")
    if sep:
        score = _parse_score(tail)
        if score is not None:
            return head.strip(), score
    return text, None


# --------------------------------------------------------------------------- #
# The lexicon
# --------------------------------------------------------------------------- #


class Lexicon:
    """A scored word list plus its pattern index.

    Words that are not pure ``A-Z`` after upper-casing are dropped rather than
    silently repaired: cleaning belongs in :mod:`xword.lexicon.build`, where the
    caller can count what was thrown away. Duplicates keep their best score.
    """

    __slots__ = ("_scores", "_frequencies", "_index")

    def __init__(self, entries: Mapping[str, float]) -> None:
        scores: dict[str, float] = {}
        for raw, score in entries.items():
            word = raw.strip().upper()
            if not word or not (word.isascii() and word.isalpha()):
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value != value:
                continue
            value = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
            if value > scores.get(word, -1.0):
                scores[word] = value

        self._scores = scores
        self._frequencies: dict[str, int] = {}
        self._index: PatternIndex | None = None

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_entries(cls, entries: Iterable[LexiconEntry]) -> Lexicon:
        """Build from :class:`LexiconEntry` objects, keeping their frequencies."""
        items = list(entries)
        lexicon = cls({e.word: e.score for e in items})
        for entry in items:
            word = entry.word.strip().upper()
            if entry.frequency and word in lexicon._scores:
                lexicon._frequencies[word] = max(
                    entry.frequency, lexicon._frequencies.get(word, 0)
                )
        return lexicon

    @classmethod
    def empty(cls) -> Lexicon:
        """A lexicon with no words. Matching always returns nothing."""
        return cls({})

    @classmethod
    def fallback(cls) -> Lexicon:
        """The small built-in list, for when no lexicon file has been built."""
        global _FALLBACK_CACHE
        if _FALLBACK_CACHE is None:
            _FALLBACK_CACHE = cls(BUILTIN_FALLBACK)
        return _FALLBACK_CACHE

    @classmethod
    def default(cls) -> Lexicon:
        """The configured lexicon, or the built-in fallback if it is missing.

        Falling back rather than raising is deliberate: a fresh checkout has no
        ``data/lexicon/lexicon.txt``, and the package still has to import and
        the tests still have to run with no network and no build step. The
        fallback is far too small to solve a real puzzle -- run
        :func:`xword.lexicon.build.build_default_lexicon` for that.
        """
        global _DEFAULT_CACHE
        path = Path(config.DEFAULT_LEXICON_PATH)
        try:
            stat = path.stat()
        except OSError:
            return cls.fallback()

        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if _DEFAULT_CACHE is not None and _DEFAULT_CACHE[0] == key:
            return _DEFAULT_CACHE[1]
        lexicon = cls.load(path)
        _DEFAULT_CACHE = (key, lexicon)
        return lexicon

    @classmethod
    def load(cls, path: str | Path) -> Lexicon:
        """Read a ``WORD;score`` file. See :func:`parse_score_line` for what else
        is tolerated."""
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        scores: dict[str, float] = {}
        for line in text.splitlines():
            parsed = parse_score_line(line)
            if parsed is None:
                continue
            word, score = parsed
            scores[word] = DEFAULT_LOADED_SCORE if score is None else score
        return cls(scores)

    def save(self, path: str | Path) -> None:
        """Write ``WORD;score`` lines, word-sorted, UTF-8, LF endings.

        Sorted and fixed-precision so that two builds of the same inputs produce
        byte-identical files and a diff shows what actually changed.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(
            f"{word};{self._scores[word]:.6f}\n" for word in sorted(self._scores)
        )
        target.write_text(body, encoding="utf-8", newline="\n")

    # -- queries ----------------------------------------------------------- #

    @property
    def index(self) -> PatternIndex:
        """The pattern index, built on first use and then reused.

        Lazy because loading a lexicon is cheap and indexing it is not, and
        plenty of callers (the ablation that disables the lexicon, anything that
        only wants :meth:`score`) never match a pattern at all.
        """
        if self._index is None:
            words = list(self._scores)
            self._index = PatternIndex(words, [self._scores[w] for w in words])
        return self._index

    def match(self, pattern: str, limit: int = 100) -> list[tuple[str, float]]:
        """Words fitting ``pattern`` (``?`` is a wildcard), best score first."""
        return self.index.match(pattern, limit)

    def count(self, pattern: str) -> int:
        """How many words fit ``pattern``."""
        return self.index.count(pattern)

    def letters_at(self, pattern: str, position: int) -> dict[str, float]:
        """Score mass per letter at ``position`` across everything matching."""
        return self.index.letters_at(pattern, position)

    def score(self, word: str) -> float:
        """Quality of ``word`` in ``[0, 1]``; ``0.0`` if it is not in the list."""
        return self._scores.get(word.strip().upper(), 0.0)

    def frequency(self, word: str) -> int:
        """How many puzzles ``word`` was mined from, or ``0`` if unknown."""
        return self._frequencies.get(word.strip().upper(), 0)

    def entries(self) -> tuple[LexiconEntry, ...]:
        """Every word as a :class:`LexiconEntry`, word-sorted."""
        return tuple(
            LexiconEntry(word, self._scores[word], self._frequencies.get(word, 0))
            for word in sorted(self._scores)
        )

    def scores(self) -> dict[str, float]:
        """A copy of the underlying word -> score mapping."""
        return dict(self._scores)

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        return word.strip().upper() in self._scores

    def __len__(self) -> int:
        return len(self._scores)

    def __iter__(self) -> Iterator[str]:
        return iter(self._scores)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Lexicon({len(self._scores)} words)"


_DEFAULT_CACHE: tuple[tuple[str, int, int], Lexicon] | None = None
_FALLBACK_CACHE: Lexicon | None = None


# --------------------------------------------------------------------------- #
# Built-in fallback
# --------------------------------------------------------------------------- #
#
# Enough common fill that the package imports, the tests run, and a toy grid can
# actually be filled with no data directory and no network. Tiered by how
# willingly a constructor would use the word, which is what the score means.

_FALLBACK_TIERS: tuple[tuple[float, str], ...] = (
    (
        0.95,
        """
        ERA ERE ERR ERG ETA ETUI EPEE OLEO OREO ALOE ANTE ARIA ASEA ALEE AREA
        ACRE ANEW AVER IDEA IRE ODE ODES EDEN ELSE EASE EELS AGES AIDE ALES
        AMEN ANTI APEX AURA AUTO ARID EDIT EMIT EPIC IOTA ISLE ITEM OBOE OMIT
        ONCE ONTO OPAL OPEN ORAL OVAL EAVE ETCH OATS ONES EROS ARES ALTO ADOS
        """,
    ),
    (
        0.85,
        """
        ACE ACT ADD ADO AFT AGE AGO AID AIM AIR ALE ALL AMP AND ANT APE APT ARC
        ARE ARK ARM ART ASH ASK ATE AWE AXE BAD BAG BAN BAR BAT BAY BED BEE BEG
        BET BID BIG BIN BIT BOA BOG BOW BOX BOY BUD BUG BUN BUS BUY CAB CAN CAP
        CAR CAT COB COD COG CON COO COP COT COW CRY CUB CUE CUP CUT DAB DAM DAY
        DEN DEW DID DIE DIG DIM DIN DIP DOE DOG DON DOT DRY DUE DUG DUO DYE EAR
        EAT EBB EEL EGG EGO ELF ELK ELM END EON EVE EWE EYE FAD FAN FAR FAT FED
        FEE FEW FIB FIG FIN FIR FIT FIX FLU FLY FOE FOG FOR FOX FRY FUN FUR GAL
        GAP GAS GEL GEM GET GIG GIN GNU GOD GOO GOT GUM GUN GUT GUY GYM HAD HAM
        HAS HAT HAY HEM HEN HER HEW HEX HID HIM HIP HIS HIT HOE HOG HOP HOT HOW
        HUB HUE HUG HUM HUT ICE ICY ILK ILL IMP INK INN ION IRK IVY JAB JAM JAR
        JAW JAY JET JIG JOB JOG JOT JOY JUG KEG KEY KID KIN KIT LAB LAD LAG LAP
        LAW LAX LAY LEA LED LEE LEG LET LID LIE LIP LIT LOB LOG LOT LOW LUG LYE
        MAD MAN MAP MAR MAT MAW MAY MEN MET MID MIX MOB MOD MOM MOP MOW MUD MUG
        NAB NAG NAP NAY NET NEW NIL NIP NIT NOD NOR NOT NOW NUN NUT OAF OAK OAR
        OAT ODD OFF OFT OHM OIL OLD ONE OPT ORB ORE OUR OUT OVA OWE OWL OWN PAD
        PAL PAN PAR PAT PAW PAY PEA PEG PEN PEP PER PET PEW PIE PIG PIN PIT PLY
        POD POP POT PRO PRY PUB PUN PUP PUT RAG RAM RAN RAP RAT RAW RAY RED REF
        RIB RID RIG RIM RIP ROB ROD ROE ROT ROW RUB RUE RUG RUM RUN RUT RYE SAD
        SAG SAP SAT SAW SAY SEA SEE SET SEW SHE SHY SIN SIP SIR SIT SIX SKI SKY
        SLY SOB SOD SON SOW SOY SPA SPY STY SUE SUM SUN TAB TAD TAG TAN TAP TAR
        TAX TEA TEE TEN THE TIC TIE TIN TIP TOE TON TOO TOP TOT TOW TOY TRY TUB
        TUG TWO URN USE VAN VAT VET VEX VIA VIE VOW WAD WAG WAR WAS WAX WAY WEB
        WED WEE WET WHO WHY WIG WIN WIT WOE WOK WON WOO WRY YAK YAM YAP YEA YEN
        YES YET YEW YOU ZAP ZIP ZOO
        """,
    ),
    (
        0.75,
        """
        ABLE ACID AGED ALSO ARCH ARMY ATOM AUNT AWAY BABY BACK BAKE BALD BALL
        BAND BANK BARE BARN BASE BATH BEAD BEAM BEAN BEAR BEAT BEEN BEER BELL
        BELT BEND BENT BEST BIKE BILL BIND BIRD BITE BLOW BLUE BOAT BODY BOIL
        BOLD BOLT BOND BONE BOOK BOOM BOOT BORE BORN BOSS BOTH BOWL BRED BREW
        BROW BULK BULL BURN BURY BUSH BUSY CAFE CAGE CAKE CALF CALL CALM CAME
        CAMP CANE CAPE CARD CARE CART CASE CASH CAST CAVE CELL CENT CHAT CHEF
        CHEW CHIN CHIP CHOP CITE CITY CLAM CLAN CLAP CLAW CLAY CLIP CLUB CLUE
        COAL COAT CODE COIL COIN COLD COLT COMB COME CONE COOK COOL COPE COPY
        CORD CORE CORK CORN COST COZY CREW CROP CROW CUBE CURB CURE CURL DARE
        DARK DART DASH DATA DATE DAWN DEAL DEAN DEAR DEBT DECK DEED DEEP DEER
        DENT DESK DIAL DIET DIME DINE DIRT DISH DIVE DOCK DOME DONE DOOR DOSE
        DOVE DOWN DRAG DRAW DREW DRIP DROP DRUM DUAL DUCK DUEL DULL DUMP DUST
        DUTY EACH EARL EARN EAST EASY ECHO FACE FACT FADE FAIL FAIR FAKE FALL
        FAME FARE FARM FAST FATE FEAR FEAT FEED FEEL FEET FELL FELT FERN FEUD
        FILE FILL FILM FIND FINE FIRE FIRM FISH FIST FIVE FLAG FLAT FLED FLEE
        FLEW FLIP FLOW FOAM FOLD FOLK FOND FOOD FOOL FOOT FORD FORE FORK FORM
        FORT FOUL FOUR FREE FROG FROM FUEL FULL FUND FUSE GAIN GAME GATE GAVE
        GEAR GENE GIFT GIRL GIVE GLAD GLOW GOAL GOAT GOES GOLD GOLF GONE GOOD
        GOWN GRAB GRAY GREW GRID GRIM GRIN GRIP GROW GULF HAIL HAIR HALF HALL
        HALT HAND HANG HARD HARE HARM HARP HASH HATE HAUL HAVE HAWK HEAD HEAL
        HEAP HEAR HEAT HEED HEEL HEIR HELD HELP HERB HERD HERE HERO HIDE HIGH
        HIKE HILL HINT HIRE HIVE HOLD HOLE HOLY HOME HOOD HOOK HOPE HORN HOSE
        HOST HOUR HUGE HUNT HURT ICON IDLE INCH INTO IRON JADE JAIL JAZZ JOIN
        JOKE JUMP JUNE JURY JUST KEEN KEEP KELP KEPT KICK KILN KIND KING KISS
        KITE KNEE KNEW KNIT KNOB KNOT KNOW LACE LACK LADY LAID LAKE LAMB LAME
        LAMP LAND LANE LAST LATE LAVA LAWN LAZY LEAD LEAF LEAK LEAN LEAP LEFT
        LEND LENS LENT LESS LIAR LIFE LIFT LIKE LIMB LIME LINE LINK LION LIST
        LIVE LOAD LOAF LOAN LOCK LOFT LOGO LONE LONG LOOK LOOM LOOP LORD LOSE
        LOSS LOST LOUD LOVE LUCK LUNG LURE LUSH MADE MAID MAIL MAIN MAKE MALE
        MALL MANE MANY MARE MARK MASK MAST MATE MATH MAZE MEAL MEAN MEAT MEET
        MELT MEMO MEND MENU MERE MESH MICE MILD MILE MILK MILL MIND MINE MINT
        MISS MIST MOAT MODE MOLD MOLE MONK MOOD MOON MORE MOSS MOST MOTH MOVE
        MUCH MULE MUTE NAIL NAME NAVY NEAR NEAT NECK NEED NEON NEST NEWS NEXT
        NICE NINE NODE NOON NOSE NOTE NOUN OKAY ONLY OOZE OVEN OVER PACE PACK
        PAGE PAID PAIL PAIN PAIR PALE PALM PANE PARK PART PASS PAST PATH PAVE
        PEAK PEAR PEAT PECK PEEL PEER PELT PERK PEST PICK PIER PIKE PILE PILL
        PINE PINK PINT PIPE PITY PLAN PLAY PLEA PLOT PLOW PLUG PLUM PLUS POEM
        POET POLE POLL POND PONY POOL POOR PORE PORK PORT POSE POST POUR PRAY
        PREP PREY PROP PULL PULP PUMP PURE PUSH QUIT QUIZ RACE RACK RAFT RAGE
        RAID RAIL RAIN RAKE RAMP RANG RANK RANT RARE RASH RATE RAVE READ REAL
        REAP REAR REED REEF REEL REIN RELY REND RENT REST RIBS RICE RICH RIDE
        RIFT RING RINK RIOT RIPE RISE RISK RITE ROAD ROAM ROAR ROBE ROCK RODE
        ROLE ROLL ROOF ROOM ROOT ROPE ROSE ROSY RUDE RUIN RULE RUNG RUSH RUST
        SAFE SAGA SAGE SAID SAIL SAKE SALE SALT SAME SAND SANE SANG SANK SAVE
        SCAN SCAR SEAL SEAM SEAT SEED SEEK SEEM SEEN SEIZE SELF SELL SEND SENT
        SHED SHIP SHOE SHOP SHOT SHOW SHUT SICK SIDE SIGH SIGN SILK SILO SING
        SINK SITE SIZE SKIN SKIP SLAB SLAM SLAP SLED SLEW SLID SLIM SLIP SLOT
        SLOW SNAP SNOW SOAK SOAP SOAR SOCK SODA SOFA SOFT SOIL SOLD SOLE SOLO
        SOME SONG SOON SORE SORT SOUL SOUP SOUR SPAN SPAR SPIN SPIT SPOT SPUR
        STAB STAG STAR STAY STEM STEP STEW STIR STOP STOW STUB STUN SUCH SUIT
        SUNG SUNK SURE SURF SWAN SWAP SWAY SWIM TACK TAIL TAKE TALE TALK TALL
        TAME TANK TAPE TART TASK TEAM TEAR TEEM TELL TEND TENT TERM TEST TEXT
        THAN THAT THAW THEM THEN THEY THIN THIS THUD THUS TIDE TIDY TIED TIER
        TILE TILL TILT TIME TINY TIRE TOAD TOLD TOLL TOMB TONE TOOK TOOL TORE
        TORN TOSS TOUR TOWN TRAM TRAP TRAY TREE TREK TRIM TRIO TRIP TROT TRUE
        TUBE TUCK TUNA TUNE TURF TURN TWIN TWIST TYPE UGLY UNDO UNIT UPON URGE
        USED USER VAIN VASE VAST VEAL VEIL VEIN VERB VERY VEST VIEW VINE VISA
        VOID VOLT VOTE WADE WAGE WAIT WAKE WALK WALL WAND WANE WANT WARD WARM
        WARN WARP WART WASH WASP WAVE WEAK WEAR WEED WEEK WEEP WELD WELL WENT
        WEPT WERE WEST WHAT WHEN WHIM WHIP WHOM WIDE WIFE WILD WILL WIND WINE
        WING WINK WIPE WIRE WISE WISH WITH WOKE WOLF WOOD WOOL WORD WORE WORK
        WORM WORN WRAP YARD YARN YEAR YELL YOGA YOLK ZERO ZINC ZONE
        """,
    ),
    (
        0.6,
        """
        ABOUT ABOVE ADAPT ADMIT ADOPT AFTER AGAIN AGENT AGREE AHEAD ALARM ALBUM
        ALIVE ALLOW ALONE ALONG ALTER AMBER AMEND AMONG ANGEL ANGER ANGLE ANKLE
        APART APPLE APRIL ARENA ARGUE ARISE ARROW ASIDE ASSET AUDIO AVOID AWAKE
        AWARD AWARE BACON BADGE BAKER BASIC BASIL BEACH BEARD BEAST BEGAN BEGIN
        BEING BELOW BENCH BERRY BIRTH BLACK BLADE BLAME BLANK BLAST BLAZE BLEND
        BLIND BLOCK BLOOD BLOOM BOARD BOAST BONUS BOOST BOOTH BOUND BRAIN BRAKE
        BRAND BRAVE BREAD BREAK BRICK BRIDE BRIEF BRING BROAD BROOK BROOM BROWN
        BRUSH BUILD BUNCH BURST CABIN CABLE CANAL CANDY CANOE CARGO CAROL CARRY
        CARVE CATCH CAUSE CEDAR CHAIN CHAIR CHALK CHARM CHART CHASE CHEAP CHECK
        CHEEK CHEER CHESS CHEST CHIEF CHILD CHILI CHILL CHOIR CHORE CHOSE CIDER
        CIGAR CIVIC CLAIM CLAMP CLASH CLASS CLEAN CLEAR CLERK CLIFF CLIMB CLOCK
        CLOSE CLOTH CLOUD CLOWN COACH COAST COCOA COMET COMIC CORAL COUCH COUGH
        COULD COUNT COURT COVER CRACK CRAFT CRANE CRASH CRATE CRAWL CRAZY CREAM
        CREEK CREPT CREST CRIME CRISP CROSS CROWD CROWN CRUDE CRUEL CRUMB CRUSH
        CURVE CYCLE DAILY DAIRY DANCE DATED DEALT DEBUT DECAY DECOR DELAY DELTA
        DEPTH DERBY DEVIL DIARY DIRTY DITCH DIVER DODGE DOING DOUBT DOZEN DRAFT
        DRAIN DRAMA DRANK DREAM DRESS DRIED DRIFT DRILL DRINK DRIVE DROVE DROWN
        DRUNK DUSTY DWELL EAGER EAGLE EARLY EARTH EIGHT ELBOW ELDER ELECT ELITE
        EMPTY ENACT ENEMY ENJOY ENTER ENTRY EQUAL ERROR ESSAY EVENT EVERY EXACT
        EXCEL EXIST EXTRA FABLE FAINT FAITH FALSE FANCY FARCE FATAL FAULT FAVOR
        FEAST FENCE FERRY FETCH FEVER FIBER FIELD FIERY FIFTH FIFTY FIGHT FINAL
        FIRST FLAME FLASH FLEET FLESH FLICK FLINT FLOAT FLOCK FLOOD FLOOR FLOUR
        FLUID FLUSH FOCUS FORCE FORGE FORTH FORTY FORUM FOUND FRAME FRANK FRAUD
        FRESH FRONT FROST FROWN FRUIT FUDGE FUNNY GAUGE GHOST GIANT GLAND GLARE
        GLASS GLEAM GLIDE GLOBE GLORY GLOVE GRACE GRADE GRAIN GRAND GRANT GRAPE
        GRAPH GRASP GRASS GRAVE GRAVY GREAT GREEN GREET GRIEF GRILL GRIND GROOM
        GROVE GUARD GUESS GUEST GUIDE GUILD GUILT HABIT HANDY HAPPY HARSH HASTE
        HATCH HEARD HEART HEAVY HEDGE HELLO HENCE HOBBY HONEY HONOR HORSE HOTEL
        HOUSE HUMAN HUMID HUMOR HURRY IDEAL IMAGE IMPLY INDEX INNER INPUT ISSUE
        IVORY JELLY JEWEL JOINT JOKER JUDGE JUICE KNIFE KNOCK KNOWN LABEL LABOR
        LARGE LASER LATER LAUGH LAYER LEARN LEASE LEAST LEAVE LEDGE LEGAL LEMON
        LEVEL LEVER LIGHT LIMIT LINEN LIVER LOBBY LOCAL LODGE LOGIC LOOSE LOWER
        LOYAL LUCKY LUNAR LUNCH LYRIC MAGIC MAJOR MAPLE MARCH MARSH MATCH MAYBE
        MAYOR MEDAL MEDIA MELON MERCY MERGE MERIT MERRY METAL METER MIDST MIGHT
        MINOR MINUS MIXER MODEL MOIST MONEY MONTH MORAL MOTOR MOUNT MOUSE MOUTH
        MOVIE MUSIC NAIVE NASAL NAVAL NERVE NEVER NEWLY NIGHT NINTH NOBLE NOISE
        NORTH NOTED NOVEL NURSE OCCUR OCEAN OFFER OFTEN OLIVE ONION ONSET OPERA
        ORBIT ORDER ORGAN OTHER OTTER OUGHT OUNCE OUTER OWNER OXIDE PAINT PANEL
        PANIC PAPER PARTY PASTA PASTE PATCH PATIO PAUSE PEACE PEACH PEARL PEDAL
        PENNY PERCH PERIL PHASE PHONE PHOTO PIANO PIECE PILOT PINCH PITCH PIVOT
        PIXEL PIZZA PLACE PLAID PLAIN PLANE PLANT PLATE PLAZA PLEAD PLUCK PLUMB
        POINT POLAR PORCH POUND POWER PRESS PRICE PRIDE PRIME PRINT PRIOR PRIZE
        PROBE PRONE PROOF PROSE PROUD PROVE PRUNE PULSE PUNCH PUPIL PUREE PURSE
        QUART QUEEN QUERY QUEST QUEUE QUICK QUIET QUILT QUOTE RADAR RADIO RAISE
        RALLY RANCH RANGE RAPID RATIO RAVEN REACH READY REALM REBEL REFER REIGN
        RELAX RELAY REPLY RIDER RIDGE RIFLE RIGHT RIGID RINSE RISKY RIVAL RIVER
        ROAST ROBIN ROBOT ROCKY ROGUE ROMAN ROOST ROTOR ROUGH ROUND ROUTE ROYAL
        RUGBY RULER RUMOR RURAL SADLY SAINT SALAD SALON SALSA SANDY SAUCE SAUNA
        SCALE SCARF SCENE SCENT SCOLD SCOOP SCOPE SCORE SCOUT SCRAP SCREW SCRUB
        SEDAN SEIZE SENSE SERVE SEVEN SHADE SHAFT SHAKE SHALE SHALL SHAME SHAPE
        SHARE SHARK SHARP SHAVE SHEAR SHEEP SHEER SHEET SHELF SHELL SHIFT SHINE
        SHIRT SHOCK SHONE SHOOK SHOOT SHORE SHORT SHOUT SHOVE SHOWN SHRUB SIEGE
        SIGHT SILLY SINCE SIREN SIXTH SIXTY SKATE SKIRT SKULL SLANT SLATE SLEEP
        SLEET SLEPT SLICE SLIDE SLOPE SMALL SMART SMASH SMELL SMILE SMOKE SNACK
        SNAKE SNEAK SNIFF SOLAR SOLID SOLVE SORRY SOUND SOUTH SPACE SPADE SPARE
        SPARK SPEAK SPEAR SPEED SPELL SPEND SPENT SPICE SPIKE SPILL SPINE SPIRE
        SPLIT SPOKE SPOON SPORT SPRAY SQUAD STACK STAFF STAGE STAIN STAIR STAKE
        STALE STALK STALL STAMP STAND STARE START STATE STEAK STEAL STEAM STEEL
        STEEP STEER STERN STICK STIFF STILL STING STOCK STOLE STONE STOOD STOOL
        STOOP STORE STORM STORY STOVE STRAP STRAW STRAY STRIP STUCK STUDY STUFF
        STUMP STUNT STYLE SUGAR SUITE SUNNY SUPER SURGE SWAMP SWARM SWEAR SWEAT
        SWEEP SWEET SWELL SWEPT SWIFT SWING SWORD SWORE TABLE TAKEN TALON TANGO
        TASTE TEACH TEASE TEETH TEMPO TENOR TENSE TENTH THANK THEFT THEIR THEME
        THERE THESE THICK THIEF THIGH THING THINK THIRD THORN THOSE THREE THREW
        THROW THUMB TIGER TIGHT TIMER TIMID TITLE TOAST TODAY TOKEN TOOTH TOPIC
        TORCH TOTAL TOUCH TOUGH TOWEL TOWER TOXIC TRACE TRACK TRACT TRADE TRAIL
        TRAIN TRAIT TRAMP TRASH TREAT TREND TRIAL TRIBE TRICK TRIED TROOP TROUT
        TRUCK TRULY TRUNK TRUST TRUTH TULIP TUTOR TWICE TWINE TWIST ULCER UNCLE
        UNDER UNION UNITE UNTIL UPPER UPSET URBAN USAGE USHER USUAL UTTER VAGUE
        VALID VALUE VALVE VAPOR VAULT VENUE VERGE VERSE VIDEO VIGIL VILLA VINYL
        VIOLA VIRUS VISIT VITAL VIVID VOCAL VOICE VOTER VOWEL WAGER WAGON WAIST
        WALTZ WASTE WATCH WATER WEARY WEAVE WEDGE WEIGH WEIRD WHALE WHEAT WHEEL
        WHERE WHICH WHILE WHITE WHOLE WHOSE WIDOW WIDTH WINCE WITCH WOMAN WOMEN
        WORLD WORRY WORSE WORST WORTH WOULD WOUND WOVEN WRECK WRIST WRITE WRONG
        WROTE YACHT YEARN YEAST YIELD YOUNG YOUTH ZEBRA ZESTY
        """,
    ),
    (
        0.5,
        """
        ABROAD ACCEPT ACCESS ACCUSE ACROSS ACTION ACTIVE ACTUAL ADVICE ADVISE
        AFFECT AFFORD AFRAID AGENDA ALMOND ALMOST ALWAYS AMOUNT ANCHOR ANIMAL
        ANNUAL ANSWER ANTHEM ANYONE ANYWAY APPEAL APPEAR APPLES ARCADE ARCHER
        ARMORY ARRIVE ARTIST ASLEEP ASSIST ASSUME ATTACH ATTACK ATTEND AUTUMN
        AVENUE BABBLE BACKUP BALLAD BALLET BANANA BANNER BARREL BASKET BATTER
        BATTLE BEACON BEAUTY BECAME BECOME BEFORE BEHALF BEHAVE BEHIND BELIEF
        BELONG BENEATH BESIDE BETRAY BETTER BEYOND BICYCLE BINARY BIRTHDAY
        BISHOP BITTER BLAZER BLOSSOM BOTTLE BOTTOM BOUNCE BRANCH BRANDY BREACH
        BREATH BREEZE BRIDGE BRIGHT BRONZE BROKEN BROTHER BRUNCH BUCKET BUDGET
        BUFFET BUNDLE BURDEN BUREAU BUTTER BUTTON CABBAGE CABINET CAMERA CAMPUS
        CANDLE CANNON CANYON CAPTAIN CAPTURE CARBON CAREER CARPET CARROT CARTON
        CASINO CASTLE CASUAL CATTLE CAVERN CEILING CELLAR CEMENT CENTER CENTRAL
        CHAPEL CHARGE CHARITY CHERRY CHOICE CHORUS CHROME CIRCLE CIRCUS CITRUS
        CLAMOR CLASSIC CLIMATE CLOSET CLOTHES CLOVER CLUSTER COFFEE COLLAR
        COLLEGE COLUMN COMBAT COMEDY COMFORT COMMON COMPANY CONCERT CONDUCT
        CONFIRM CONNECT CONSENT CONTENT CONTEST CONTROL CONVERT COOKIE COPPER
        CORNER CORRECT COTTAGE COTTON COUGAR COUNCIL COUNTER COUNTRY COUPLE
        COURAGE COURSE COUSIN COVERT CRADLE CRAYON CREATE CREDIT CRICKET CRISIS
        CRITIC CRUISE CRYSTAL CULTURE CURFEW CURIOUS CURRENT CURTAIN CUSTOM
        DAMAGE DANGER DEALER DEBATE DECADE DECENT DECIDE DECLARE DECLINE DEFEAT
        DEFEND DEFINE DEGREE DELETE DELIVER DEMAND DENIAL DENTAL DEPART DEPEND
        DEPOSIT DESERT DESIGN DESIRE DETAIL DETECT DEVICE DEVOTE DIALECT DIAMOND
        DIGEST DINNER DIRECT DISARM DISCUSS DISPLAY DISTANT DIVIDE DOCTOR DOLLAR
        DONKEY DOUBLE DRAGON DRAWER DREAMY DRIVER DROUGHT DYNAMIC EASTER ECLIPSE
        EDITOR EFFECT EFFORT ELEVEN EMPIRE EMPLOY ENABLE ENCORE ENGAGE ENGINE
        ENOUGH ENSURE ENTIRE EQUATOR ESCAPE ESTATE ETHNIC EVENING EXCEED EXCEPT
        EXCUSE EXHALE EXPAND EXPECT EXPERT EXPIRE EXPORT EXTEND FABRIC FACTOR
        FAILED FAMILY FAMOUS FARMER FATHER FAVOUR FEEDER FELLOW FEMALE FIGURE
        FILTER FINGER FINISH FIREFLY FISCAL FITTED FLIGHT FLOWER FOLDER FOLLOW
        FOREST FORGET FORGIVE FORMAL FORMAT FORMER FOSTER FRIDGE FRIEND FRINGE
        FROZEN FUTURE GADGET GALAXY GALLON GARAGE GARDEN GARLIC GATHER GENIUS
        GENTLE GENUINE GERMAN GIGGLE GINGER GLACIER GLOBAL GOLDEN GOSPEL GOVERN
        GRAVEL GROUND GROUPS GROWTH GUITAR HALLWAY HAMMER HAMPER HANDLE HAPPEN
        HARBOR HARVEST HAZARD HEALTH HEARTY HEAVEN HEIGHT HELMET HERALD HIDDEN
        HOLIDAY HOLLOW HONEST HORROR HOSTEL HUMBLE HUNGER HUNTER HURDLE HUSBAND
        HYBRID ICONIC IMPACT IMPORT INCOME INDIGO INDOOR INFANT INFORM INJURE
        INSECT INSIDE INSIST INSPIRE INSTEAD INTEND INVENT INVEST INVITE ISLAND
        ITSELF JACKET JAGUAR JOURNAL JOURNEY JUNGLE JUNIOR JUSTICE KETTLE KINDLE
        KITCHEN KNIGHT LADDER LANTERN LAPTOP LATTER LAUNCH LAUNDRY LAWYER LEADER
        LEAGUE LEATHER LECTURE LEGACY LEGEND LEMONADE LENGTH LESSON LETTER
        LIBRARY LICENSE LIKELY LINGER LIQUID LISTEN LITTLE LIVELY LOCKET LODGER
        LONELY LOUNGE LOVELY LOWEST LOYALTY LUXURY MAGNET MAIDEN MAMMAL MANAGE
        MANNER MANUAL MARBLE MARGIN MARINE MARKET MARVEL MASTER MATTER MATURE
        MEADOW MEDIUM MEMBER MEMORY MENTAL MENTOR MERCURY MERGER METHOD MIDDAY
        MIDDLE MIGHTY MINGLE MINUTE MIRROR MISERY MISSION MOBILE MODERN MODEST
        MODULE MOMENT MONKEY MONTHLY MORNING MORTAL MOSAIC MOSTLY MOTHER MOTION
        MOTIVE MOUNTAIN MURMUR MUSCLE MUSEUM MUSTARD MUTUAL MYSTERY NARROW
        NATION NATIVE NATURE NEARBY NEATLY NECTAR NEEDLE NEPHEW NETWORK NEUTRAL
        NIBBLE NICKEL NOBODY NOTICE NOTION NOVICE NUMBER NURSERY OBJECT OBLIGE
        OBSERVE OBTAIN OCCUPY OCTOBER OFFICE OFFSET ONWARD OPPOSE ORANGE ORCHID
        ORDEAL ORIGIN ORPHAN OUTAGE OUTFIT OUTLET OUTLINE OUTPUT OUTSIDE OVERLAP
        OXYGEN OYSTER PACKET PADDLE PALACE PANDA PANTHER PARADE PARCEL PARDON
        PARENT PARISH PARLOR PARROT PARTLY PASSAGE PASSION PASTOR PATENT PATROL
        PATTERN PEANUT PEBBLE PENCIL PEOPLE PEPPER PERIOD PERMIT PERSON PHRASE
        PICKLE PICNIC PIGEON PILLAR PILLOW PIRATE PISTOL PLANET PLASTER PLATEAU
        PLEASE PLENTY PLIGHT PLUNGE POCKET POETRY POLICE POLICY POLISH POLITE
        PONDER POPLAR PORTAL POTATO POTTER POUNCE POWDER PRAISE PRAYER PREACH
        PREFER PREPARE PRESENT PRETTY PREVENT PRINCE PRISON PRIVATE PROBLEM
        PROCESS PRODUCE PROFIT PROGRAM PROJECT PROMISE PROMPT PROTECT PROTEST
        PROVIDE PUBLIC PUDDLE PUPPET PURPLE PURSUE PUZZLE QUAINT QUARRY QUARTER
        QUIVER RABBIT RACKET RADIANT RADISH RAFFLE RAGGED RANDOM RANGER RANSOM
        RAPTOR RARELY RATHER RATTLE REALLY REASON REBUILD RECALL RECEIPT RECEIVE
        RECENT RECIPE RECORD RECOVER REDUCE REFLECT REFORM REFUGE REFUND REFUSE
        REGARD REGION REGRET REGULAR REJECT RELATE RELEASE RELIEF REMAIN REMARK
        REMEDY REMIND REMOTE REMOVE RENDER RENEWAL REPAIR REPEAT REPLACE REPORT
        RESCUE RESERVE RESIDE RESIST RESORT RESPECT RESULT RETAIL RETAIN RETIRE
        RETURN REVEAL REVERSE REVIEW REVISE REWARD RIBBON RIDDLE RIPPLE RITUAL
        RIVALS ROBBER ROCKET RODENT ROSTER ROTATE RUBBER RUBBLE RUGGED RUNNER
        RUSTIC SADDLE SAFARI SAILOR SALMON SALOON SAMPLE SANDAL SATIRE SAVAGE
        SAVIOR SAWDUST SCARCE SCHEME SCHOOL SCIENCE SCREEN SEAFOOD SEASON SECOND
        SECRET SECTOR SECURE SEEKER SELDOM SELECT SENIOR SENSOR SERIAL SERIES
        SERMON SERVANT SETTLE SEVERE SHADOW SHIELD SHIVER SHOULD SHOVEL SHOWER
        SHRIMP SHRINE SHRINK SIGNAL SILENT SILVER SIMPLE SINGER SINGLE SISTER
        SKETCH SLEEVE SLIGHT SLIPPER SLOGAN SMOOTH SNIFFLE SOCCER SOCIAL SOCKET
        SODIUM SOFTLY SOLEMN SOLVENT SOMBER SONATA SORROW SOURCE SPIRAL SPIRIT
        SPLASH SPOKEN SPONGE SPRING SPRINT SQUARE SQUEEZE STABLE STADIUM STANCE
        STANZA STAPLE STARCH STARVE STATUE STATUS STEADY STEEPLE STENCIL STICKY
        STOLEN STORAGE STRAIN STRAND STREAM STREET STRESS STRICT STRIDE STRIKE
        STRING STRIPE STROKE STRONG STRUCK STUDIO STUDENT STUPID SUBMIT SUBTLE
        SUBURB SUBWAY SUCCESS SUDDEN SUFFER SUGGEST SUMMER SUMMIT SUNDAY SUNSET
        SUPPER SUPPLY SUPPORT SURELY SURFACE SURGEON SURPLUS SURVEY SURVIVE
        SUSPECT SUSTAIN SWEATER SYMBOL SYRUP SYSTEM TACKLE TALENT TANGLE TARGET
        TARIFF TASSEL TATTOO TAVERN TEAPOT TEMPER TEMPLE TENANT TENDER TENNIS
        TERROR THANKS THEORY THIRTY THOUGH THREAD THREAT THRIVE THRONE THROUGH
        THUNDER TICKET TICKLE TIMBER TIMELY TINSEL TISSUE TOMATO TONGUE TOPPLE
        TORNADO TORRENT TOWARD TOWNSHIP TRAGIC TRAILER TRAITOR TRANSIT TRAVEL
        TREATY TREMOR TRIBUTE TRICKY TRIGGER TRIPLE TRIUMPH TROLLEY TROPHY
        TROUBLE TRUANT TRUMPET TUNNEL TURKEY TURNIP TURTLE TWELVE TWENTY TYCOON
        TYPIST UNABLE UNCOVER UNDONE UNFAIR UNIFORM UNIQUE UNITED UNLESS UNLIKE
        UNLOCK UNPACK UNREST UNTIDY UNUSED UNWIND UPDATE UPHILL UPHOLD UPLIFT
        UPROAR UPWARD URGENT USEFUL UTMOST VACANT VACUUM VALLEY VANISH VARIED
        VELVET VENDOR VENTURE VERBAL VERSUS VESSEL VETERAN VICTIM VICTORY VIEWER
        VILLAGE VINTAGE VIOLET VIOLIN VIRTUE VISION VISUAL VOLUME VOYAGE WAGGON
        WAITER WALLET WALNUT WANDER WARDEN WARMTH WARNING WARRANT WEALTH WEAPON
        WEASEL WEATHER WEEKEND WEIGHT WELCOME WELFARE WHISKER WHISPER WHISTLE
        WICKED WIDELY WILLOW WINDOW WINNER WINTER WISDOM WITHIN WITNESS WIZARD
        WONDER WOODEN WOOLEN WORKER WORTHY WRITER WRITTEN YELLOW YOGURT ZEALOUS
        ZENITH ZIGZAG
        """,
    ),
    (
        0.4,
        """
        AN AS AT BE BY DO GO HE IF IN IS IT ME MY NO OF ON OR SO TO UP US WE AM
        AH AX ID OX PI RE TI XI
        """,
    ),
)


def _build_fallback() -> dict[str, float]:
    out: dict[str, float] = {}
    for score, block in _FALLBACK_TIERS:
        for word in block.split():
            word = word.upper()
            if word.isascii() and word.isalpha() and out.get(word, 0.0) < score:
                out[word] = score
    return out


#: The offline word list. Small on purpose -- it exists so the package works in
#: a fresh checkout, not so it can solve a Saturday.
BUILTIN_FALLBACK: dict[str, float] = _build_fallback()
