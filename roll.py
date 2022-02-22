from importblob import *
from mido import MidiFile, MidiTrack, Message, MetaMessage

TAIL = 0x1
ONSET = 0x4

def bakeroll(fpath: str, quantize: int = 32) -> np.ndarray:
    midi = MidiFile(fpath)
    midi.ticks_per_beat # per 1/4

    # per 1/32 {//8} or 1/16 {//4}
    ticksatom = midi.ticks_per_beat // (quantize // 4)

    for track in midi.tracks:
        ctime = 0
        for m in track:
            ctime += round(m.time / ticksatom)

        if ctime == 0:
            continue

        grid = zeros((88, ctime+1), np.int8)
        starts = zeros(88, np.uint32)

        ctime = 0
        for m in track:
            ctime += round(m.time / ticksatom)

            if m.type == 'note_on':
                starts[-m.note] = ctime

            elif m.type == 'note_off':
                start = starts[-m.note]

                grid[-m.note, start] = ONSET
                grid[-m.note, start+1:ctime] = TAIL

        return grid

    raise ValueError(f"cannot quantize that low: {quantize} (???)")

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
                track.append(Message('note_off', note=88-note, time=dt * ticksatom))
                lasteventtime = ctime

        # open notes
        for note in findall(grid[:, ctime] == ONSET):
            if first:
                dt = ctime - lasteventtime
                first = False
            else:
                dt = 0

            opennotes[note] = True
            track.append(Message('note_on', note=88-note, time=dt * ticksatom))
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

def shardroll(grid: np.ndarray, L: int = 32, every: bool = False) -> np.ndarray:
    onsets = set(np.where(grid == ONSET)[1])
    # treat the final tick as seperate onset
    onsets.add(grid.shape[1])
    onsets = sorted(list(onsets))

    shards = []
    for sind in range(len(onsets)-1):
        if every:
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

if __name__ == '__main__':
    quantize = 32

    grid = bakeroll('../sms/opening.mid', quantize=quantize)
    m = midiroll(grid, '../sms/opening0.mid', quantize=quantize)

    grid = bakeroll('fonts/kars-break.mid', quantize=quantize)
    midiroll(grid, '../sms/kars-break0.mid', quantize=quantize)
