#/usr/bin/env python
import dynamiq_engine as dynq
import openpathsampling.engines as peng
import numpy as np

def run_engine(engine, snapshot, ntraj=1, nsteps=1000):
    for t in range(ntraj):
        engine.start()
        length_criteria = lambda x, foo : len(x) <= nsteps
        trajectory = engine.generate(snapshot, running=[length_criteria])

def morse_setup():
    # Morse test system: a test of 1D performance
    potential = dynq.potentials.interactions.MorseInteraction(D=30.0,
                                                              beta=0.08,
                                                              x0=0.0)
    integrator = dynq.integrators.CandyRozmus4(dt=0.1, potential=potential)
    masses = [1.0]
    topology = dynq.Topology(masses=[1.0], potential=potential)
    snapshot = dynq.Snapshot(coordinates=np.array([0.0]),
                             momenta=np.array([0.5]),
                             topology=topology)
    engine = dynq.DynamiqEngine(potential, integrator, template=snapshot)
    return engine, snapshot

def morse_run(ntrajs, nsteps):
    engine, snapshot = morse_setup()
    run_engine(engine, snapshot, ntrajs, nsteps)

benchmarks = {
    'morse_1_1000': ('morse_run', {'ntrajs': 1, 'nsteps': 1000}),
    'morse_100_10': ('morse_run', {'ntrajs': 100, 'nsteps': 10})
}

def argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--cprofile', action='store_true')
    parser.add_argument('--pyinstrument', action='store_true')
    return parser.parse_args()

def make_benchmark_cmd(bench):
    fcn = benchmarks[bench][0]
    args = benchmarks[bench][1]
    args_str = ", ".join([k+"="+str(args[k]) for k in args])
    cmd_str = fcn+"("+args_str+")"
    return cmd_str

import sys
if __name__ == "__main__":
    opts = argparse()
    if opts.timing:
        import timeit
        morse_run(1, 1000)
        for bench in benchmarks:
            cmd = make_benchmark_cmd(bench)
            fcn = benchmarks[bench][0]
            print bench, ":", timeit.timeit(cmd, 
                                            'from __main__ import ' + fcn, 
                                            number=5)
    if opts.cprofile:
        import cProfile
        for bench in benchmarks:
            cProfile.run(make_benchmark_cmd(bench), bench + ".pstats")
    if opts.pyinstrument:
        print "pyinstrument"
        pass
