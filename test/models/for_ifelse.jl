@model IfElseLoop begin
    for co in [H, F]
        x{co}[0] = ifelse(co == H, 1, 2)
    end
end
