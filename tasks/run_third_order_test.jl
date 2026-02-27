#!/usr/bin/env julia

# First, let's run the test with only s1 and s2 blocks
ENV["INPUT_BLOCKS"] = "s1,s2"
ENV["RUN_FD"] = "true"
ENV["RUN_FWD"] = "false"
ENV["RUN_ZYG"] = "true"
ENV["FDM_ORDER"] = "4"
ENV["FDM_MAX_RANGE"] = "1e-4"

include("compare_third_order_input_gradients_caldara.jl")
