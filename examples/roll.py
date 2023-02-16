from blob import *
from mido import MidiFile, MidiTrack, Message, MetaMessage
from itertools import cycle

TAIL = 0x1
ONSET = 0x2

def bakeroll(fpath: str, quantize: int = 32, staccato: bool = True) -> np.ndarray:
    midi = MidiFile(fpath)
    midi.ticks_per_beat # per 1/4

    # per 1/32 {//8} or 1/16 {//4}
    ticksatom = midi.ticks_per_beat // (quantize // 4)

    track = midi.tracks[np.argmax(ap(len, midi.tracks))]

    ctime = 0
    for m in track:
        ctime += round(m.time / ticksatom)

    grid = zeros((128, ctime+1), np.int8)
    starts = zeros(128, np.uint32)

    ctime = 0
    for m in track:
        ctime += round(m.time / ticksatom)

        if m.type == 'note_off' or (m.type == 'note_on' and m.velocity == 0):
            start = starts[m.note]

            grid[m.note, start] = ONSET

            if not staccato:
                grid[m.note, start+1:ctime] = TAIL

        elif m.type == 'note_on':
            starts[m.note] = ctime

    return grid

roots = ['C', 'D♭', 'D', 'E♭', 'E', 'F', 'G♭', 'G', 'A♭', 'A', 'B♭', 'B']
intervals = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']
class Note(NamedTuple):
    pitch: int
    len: int
    jump: int

    def __repr__(self):
        sign = '-' if self.jump < 0 else ''
        if abs(self.jump) < len(intervals):
            interval = intervals[abs(self.jump)]
        else:
            interval = f'${self.jump}'

        return f'({roots[self.pitch % 12]} [{self.len}] @ {sign}{interval})'

def bakeline(fpath: str, quantize: int = 32) -> np.ndarray:
    midi = MidiFile(fpath)
    midi.ticks_per_beat # per 1/4

    # per 1/32 {//8} or 1/16 {//4}
    ticksatom = midi.ticks_per_beat // (quantize // 4)

    track = midi.tracks[np.argmax(ap(len, midi.tracks))]

    ctime = 0
    for m in track:
        ctime += round(m.time / ticksatom)

    starts = zeros(128, int)
    prev = None
    out = []

    ctime = 0
    for m in track:
        ctime += round(m.time / ticksatom)

        if m.type == 'note_off' or (m.type == 'note_on' and m.velocity == 0):
            start = starts[m.note]

            if prev is None:
                prev = m.note

            mag = m.note - prev
            sign = np.sign(mag)
            out.append(Note(m.note, ctime-start, sign * abs(mag % 24)))
            prev = m.note

        elif m.type == 'note_on':
            starts[m.note] = ctime

    return out

def midiroll(grid: np.ndarray, fpath: str, quantize: int = 32) -> MidiFile:
    opennotes = zeros(88, bool)
    length = grid.shape[-1]

    midi = MidiFile()
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='reveling', time=0))

    ticksatom = midi.ticks_per_beat // (quantize // 4)

    lasteventtime = 0
    for ctime in range(grid.shape[-1]):
        first = True
        # close already opened notes
        for note in findall(opennotes):
            if grid[note, ctime] == 0 or grid[note, ctime] == ONSET:
                if first:
                    dt = ctime - lasteventtime
                    first = False
                else:
                    dt = 0

                opennotes[note] = False
                track.append(Message('note_off', note=note, time=dt * ticksatom))
                lasteventtime = ctime

        # open notes
        for note in findall(grid[:, ctime] == ONSET):
            if first:
                dt = ctime - lasteventtime
                first = False
            else:
                dt = 0

            opennotes[note] = True
            track.append(Message('note_on', note=note, time=dt * ticksatom))
            lasteventtime = ctime

    midi.tracks.append(track)
    midi.save(fpath)
    return midi

def forceunique(xs: np.ndarray) -> np.ndarray:
    unique = []
    for ind in range(len(xs)):
        isunique = True
        for x in unique:
            if np.all(xs[ind] == x):
                isunique = False
                break

        if isunique:
            unique.append(xs[ind])

    return np.stack(unique)

def wholeshards(roll: np.ndarray, maxlen: int = 30) -> np.ndarray:
    "straight up cuts with leading space"
    totalen = roll.shape[1]
    shards = []
    for sind in range(totalen):
        for eind in range(sind+1, min(totalen, sind+maxlen)):
            shard = roll[:, sind:eind]
            if shard.shape[1] < maxlen:
                shard = np.pad(shard, [(0, 0), (0, maxlen - (eind - sind) - 1)])

            shards.append(shard)

    return forceunique(shards)

def shardroll(grid: np.ndarray, L: int = 32, every: bool = False) -> np.ndarray:
    onsets = set(np.where(grid == ONSET)[1])
    # treat the final tick as a seperate onset
    onsets.add(grid.shape[1])
    onsets = sorted(list(onsets))

    shards = []
    for sind in range(len(onsets)-1):
        if every:
            lastind = findfirst(lambda x: x > onsets[sind] + L, onsets) or len(onsets)
            for eind in range(sind+1, lastind):
                start = onsets[sind]
                end = onsets[eind]

                ax = grid[:, start:end]

                if ax.shape[1] < L:
                    ax = np.pad(ax, [(0, 0), (0, L - (end - start))])

                shards.append(ax)
        else:
            eind = findfirst(lambda x: x > onsets[sind] + L, onsets) or len(onsets)
            start = onsets[sind]
            end = onsets[eind-1]

            ax = grid[:, start:end]

            if ax.shape[1] < L:
                ax = np.pad(ax, [(0, 0), (0, L - (end - start))])

            shards.append(ax)

    return forceunique(shards)

def rolled(fpath, length=None, shardlen=10, staccato=True, scale=False, quantize=16):
    roll = bakeroll(fpath, quantize=quantize, staccato=staccato)
    if length is not None:
        roll = roll[:, :length]

    if scale:
        HMINOR = cycle([2, 1, 2, 2, 1, 3, 1])

        hoffset = roll[:, 0].argmax()
        hoffset += 8 * 12

        inds = []
        for offset in HMINOR:
            if hoffset < 0:
                break

            if hoffset < 88:
                inds.append(hoffset)

            hoffset -= offset

        roll = roll[inds[::-1]]

    notes = np.where(roll)[0]
    lower, upper = max(0, notes.min()-2), min(88, notes.max()+2)
    horizon = np.where(roll)[1].max()+1
    roll = roll[lower:upper, :horizon]

    offset = roll[:, 0].argmax()
    shards = wholeshards(roll, maxlen=shardlen)

    return roll, offset, shards

if __name__ == '__main__':
    roll, X, offset = rolled('tunes/the-lick.mid')
    iimshow(roll)
