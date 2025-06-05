@model IfStatementLoop begin
    for co in [H, F]
        if co == H
            x{co}[0] = 1
        else
            x{co}[0] = 2
        end
    end
end
