## Module 3 — Warp Divergence

For this assignment, i created some simple mathematical functions across 3  paths, and then ran different kernels/functions for taking the paths. 

1. Uniform: all threads take the same path
2. Interleave: each thread takes interleaving paths
3. Warp Aligned: 32 threads take the same path, aligned with warp size.


I ran the following ncu profiler command to test the GPU performance on about 1M elements and display metrics:

```
ncu --metrics gpu__time_duration.sum,
smsp__thread_inst_executed_per_inst_executed.ratio,
smsp__sass_average_branch_targets_threads_uniform.pct 
  ./assignment.exe 1048576 256
```

I ran this command in the module3 folder after building the executable. 


GPU Results (from the ncu metrics)

| Kernel | Time (µs) | Threads active / instruction | Branch uniformity |
|---|---:|---:|---:|
| uniform_kernel | 77.47 | 32.00 / 32 | 100% |
| interleaved_kernel | 279.68 | 10.84 / 32 | 88.24% |
| warp_aligned_kernel | 103.14 | 32.00 / 32 | 100% |


CPU Results (timing only from the program output)

| Host function | Time (ms) |
|---|---:|
| cpu_uniform | 1246.296 |
| cpu_interleaved | 1624.347 |
| cpu_warp_aligned | 1613.564 |


## Analysis

I was able to demonstrate through this exercise both the differences in conditional branching's effect on performance between the CPU/GPU, as well as the significance of warp alignment.

In the CPU execution, we dont see a significant difference in execution time based on conditional branching or the intervals at which they happen. the slight performance difference between the uniform path and the branching path comes down to path 0s function being the cheapest one. 

In our GPU metrics, we can see that we have a significant difference in performance for the interleaved path which looks to run 3x slower than the other two. the ncu metrics also let us see that in fact with this method, we're only able to run 1/3 of our threads at a time! this shows the stalling effect. With the warped aligned kernel, we dont see this behavior because all threads in a warp are uniform and so therefore we dont have to stall. 

## Part 5

this function sets up the executions on the CPU and GPU correctly, but it will run into timing issues because of the asynch nature of running on the GPU, as well as "lazy loading" we were talking about in the module 3 discussion. when we call a kernel function, it doesnt actually start executing synchronously with the rest of the calling C program. the kernel code is run on the gpu, and our C program isnt going to wait for it to be done. "stop" will run right after and not actually give us anything meaningful. 

when i was looking into timing methods for the gpu i ran into this same issue.. i found online that there are options to get cuda events or synch it, BUT on the very first run you also have to deal with lazy loading so you need to add a warm up to the launch as well. 

For that reason, i just went with the option of using the profiler to get the most accurate data i could. the downside is that that data cant be output from the program itself, so there are pros and cons. 