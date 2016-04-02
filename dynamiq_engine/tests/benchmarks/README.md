This directory includes several scripts for benchmarking/profiling the
dynamiq engine. 

* `profiling.py`: Python script to manage running the benchmarks
* `profiling`: shell script to run all the benchmarks and convert the
  results to a useful visual form
* `clean`: simple shell script to clear output -- just `rm -f *pstats *png`.

If you have `gprof2dot` installed, then you can just run the `profiling`
shell script. I recommend `./profiling && open *png` to view all the
gprof2dot diagrams of the call behavior.

