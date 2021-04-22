import click
from datetime import datetime
from typing import List
from midiutil import MIDIFile
from pyo import *
from algorithms.genetic import generate_genome, selection_pair, single_point_crossover, mutation

BITS_PER_NOTE = 4
KEYS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
SCALES = ["major", "minorM", "dorian", "phrygian", "lydian", "mixolydian", "majorBlues", "minorBlues"]

def int_from_bits(bits: List[int]):
    return int(sum([bit * pow(2, index) for index, bit in enumerate(bits)]))


def genome_to_melody(genome, num_bars, num_notes, num_steps,
                     pauses, key, scale, root):
    notes = [genome[i * BITS_PER_NOTE:i * BITS_PER_NOTE + BITS_PER_NOTE] for i in range(num_bars * num_notes)]
    note_length = 4 / float(num_notes)
    scl = EventScale(root=key, scale=scale, first=root)

    melody = {
        "notes": [],
        "velocity": [],
        "beat": []
    }

    for note in notes:
        integer = int_from_bits(note)

        if not pauses:
            integer = int(integer % pow(2, BITS_PER_NOTE - 1))
        if integer >= pow(2, BITS_PER_NOTE - 1):
            melody["notes"] += [0]
            melody["velocity"] += [0]
            melody["beat"] += [note_length]
        else:
            if len(melody["notes"]) > 0 and melody["notes"][-1] == integer:
                melody["beat"][-1] += note_length
            else:
                melody["notes"] += [integer]
                melody["velocity"] += [127]
                melody["beat"] += [note_length]

    steps = []
    for step in range(num_steps):
        steps.append([scl[(note + step * 2) % len(scl)] for note in melody["notes"]])

    melody["notes"] = steps
    return melody


class MyInstrument(EventInstrument):
    def __init__(self, **args):
        EventInstrument.__init__(self, **args)
        self.phase = Phasor([self.freq, self.freq * 1.003])
        self.duty = Expseg([(0, 0.15), (self.dur, 0.5)], exp=4).play()
        self.osc = Compare(self.phase, self.duty, mode="<", mul=1, add=-0.5)
        self.filt = ButLP(self.osc, freq=1000, mul=self.env).out()


def genome_to_events(genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm):
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

    return [
        Events(
            instr=MyInstrument,
            midinote=EventSeq(step, occurrences=1),
            midivel=EventSeq(melody["velocity"], occurrences=1),
            beat=EventSeq(melody["beat"], occurrences=1),
            attack=0.02,
            decay=0.1,
            sustain=0.5,
            release=0.005,
            bpm=bpm
        ) for step in melody["notes"]
    ]


def fitness(genome, s, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm):
    events = genome_to_events(genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
    s.start()

    rating = input("Rating (0-5)")

    for e in events:
        e.stop()
    s.stop()
    time.sleep(1)
    rating = int(rating)
    return rating


def save_genome_to_midi(filename, genome, num_bars, num_notes, num_steps,
                        pauses, key, scale, root, bpm):
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

    if len(melody["notes"][0]) != len(melody["beat"]) or len(melody["notes"][0]) != len(melody["velocity"]):
        raise ValueError

    mf = MIDIFile(1)
    track = 0
    channel = 0
    time = 0.0
    mf.addTrackName(track, time, "Sample")
    mf.addTempo(track, time, bpm)

    for i, vel in enumerate(melody["velocity"]):
        if vel > 0:
            for step in melody["notes"]:
                mf.addNote(track, channel, step[i], time, melody["beat"][i], vel)
        time += melody["beat"][i]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        mf.writeFile(f)


@click.command()
@click.option("--num-bars", default=8, prompt='Number of bars:', type=int)
@click.option("--num-notes", default=4, prompt='Notes per bar:', type=int)
@click.option("--num-steps", default=1, prompt='Number of steps:', type=int)
@click.option("--pauses", default=True, prompt='Add Pauses?', type=bool)
@click.option("--key", default="C", prompt='Key:', type=click.Choice(KEYS, case_sensitive=False))
@click.option("--scale", default="major", prompt='Scale:', type=click.Choice(SCALES, case_sensitive=False))
@click.option("--root", default=4, prompt='Scale Root:', type=int)
@click.option("--population-size", default=10, prompt='Size of population:', type=int)
@click.option("--num-mutations", default=2, prompt='Number of mutations:', type=int)
@click.option("--mutation-probability", default=0.5, prompt='Mutations probability:', type=float)
@click.option("--bpm", default=128, prompt='Beats per minute', type=int)

def main(num_bars, num_notes, num_steps, pauses, key, scale, root,
         population_size, num_mutations, mutation_probability, bpm):
    folder = str(int(datetime.now().timestamp()))
    population = [generate_genome(num_bars * num_notes * BITS_PER_NOTE) for _ in range(population_size)]

    s = Server().boot()

    population_id = 0
    run = True
    while run:
        random.shuffle(population)

        population_fitness = [
            (genome, fitness(genome, s, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)) for genome in
            population]

        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=True)
        population = [e[0] for e in sorted_population_fitness]
        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            def fitness_lookup(genome):
                for e in population_fitness:
                    if e[0] == genome:
                        return e[1]
                return 0

            parents = selection_pair(population, fitness_lookup)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a, num=num_mutations, probability=mutation_probability)
            offspring_b = mutation(offspring_b, num=num_mutations, probability=mutation_probability)
            next_generation += [offspring_a, offspring_b]
        print(f"End of population {population_id}")

        events = genome_to_events(population[0], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        for e in events:
            e.play()
        s.start()
        input("Here is the best match:")

        s.stop()
        for e in events:
            e.stop()
        time.sleep(1)

        events = genome_to_events(population[1], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        for e in events:
            e.play()
        s.start()
        input("Here is the second best:")

        s.stop()
        for e in events:
            e.stop()
        time.sleep(1)

        print("Saving MIDI")
        for i, genome in enumerate(population):
            save_genome_to_midi(f"{folder}/{population_id}/{scale}-{key}-{i}.mid", genome, num_bars, num_notes,
                                num_steps, pauses, key, scale, root, bpm)
        print("Success")

        run = input("Continue? [y/n]") != "n"
        population = next_generation
        population_id += 1

main()
