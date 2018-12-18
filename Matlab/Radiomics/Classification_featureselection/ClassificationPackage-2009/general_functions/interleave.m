function out = interleave(vec1,vec2)

out = []; idx = 0;
while length(out) < length(vec1) + length(vec2)
    idx = idx + 1;
    if idx <= length(vec1)
        out = [out vec1(idx)];
    end
    if idx <= length(vec2)
        out = [out vec2(idx)];
    end
end

out = out(:);