function index = decodeSuperpixelIndex(pixelValue) 
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    red = uint32(pixelValue(:,:,1));
    green = uint32(pixelValue(:,:,2));
    blue = uint32(pixelValue(:,:,3));
    % "<<" is the bit-shift operator
    index = red + bitshift(green,8) + bitshift(blue,16);
end

