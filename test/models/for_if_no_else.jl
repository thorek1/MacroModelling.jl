@model IfNoElseLoop begin
    for co in [H, F]
        if co == H
            x{co}[0] = 1
        end
    end
end
