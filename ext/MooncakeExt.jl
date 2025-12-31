module MooncakeExt

# This extension is a placeholder for Mooncake support.
# Currently, _get_loglikelihood_internal does not have a custom rrule,
# so Zygote automatically differentiates through it using the rrules 
# defined for sub-functions (get_NSSS_and_parameters, calculate_jacobian, etc.).
# 
# For Mooncake support, the sub-functions would need their own Mooncake rules,
# or a custom rrule would need to be defined for _get_loglikelihood_internal.

end # module
