function [ DCh, M ] = Deconv( I, stain )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deconvolve: Deconvolution of an RGB image into its constituent stain
% channels
% 
%
% Input:
% I         - RGB input image.
% M         - (optional) Stain matrix. 
%                        (default Ruifrok & Johnston H&E matrix)
%
%
% Note: M must be an 2x3 or 3x3 matrix, where rows corrrespond to the stain
%       vectors. If only two rows are given the third is estimated as a
%       cross product of the first two.
%
%
% Output:
% DCh       - Deconvolved Channels concatatenated to form a stack. 
%             Each channel is a double in Optical Density space.
% M         - Stain matrix.
%
%
% References:
% [1] AC Ruifrok, DA Johnston. "Quantification of histochemical staining by
%     color deconvolution". Analytical & Quantitative Cytology & Histology,
%     vol.23, no.4, pp.291-299, 2001.
%
%
% Acknowledgements:
% This function is inspired by Mitko Veta's Stain Unmixing and Normalisation 
% code, which is available for download at Amida's Website:
%     http://amida13.isi.uu.nl/?q=node/69
%
%
% Example:
%           I = imread('hestain.png');
%           [ DCh, H, E, Bg, M ] = Deconvolve( I, M);
%
%
% Copyright (c) 2013, Adnan Khan
% Department of Computer Science,
% University of Warwick, UK.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run in DEMO Mode
switch stain
    case 'HE'
        M = [0.644211, 0.716556, 0.266844;
                0.092789, 0.954111, 0.283111];
    case 'HE2'
        M = [0.49015734, 0.76897085, 0.41040173;
                0.04615336, 0.8420684, 0.5373925];
    case 'H DAB'
        M = [0.650, 0.704, 0.286;
                0.268, 0.570, 0.776];
    case 'Feulgen Light Green'
        M = [0.46420921, 0.83008335, 0.30827187;
                0.94705542, 0.25373821, 0.19650764];
    case 'Giemsa'
        M = [0.834750233, 0.513556283, 0.196330403;
                0.092789, 0.954111, 0.283111];
    case 'FastRed FastBlue DAB'
        M = [0.21393921, 0.85112669, 0.47794022;
                0.74890292, 0.60624161, 0.26731082;
                0.268, 0.570, 0.776];
    case 'Methyl Green DAB'
        M = [0.98003, 0.144316, 0.133146;
                0.268, 0.570, 0.776];
    case 'H&E DAB'
        M = [0.650, 0.704, 0.286;
                0.072, 0.990, 0.105;
                0.268, 0.570, 0.776];
    case 'H AEC'
        M = [0.650, 0.704, 0.286;
                0.2743, 0.6796, 0.6803];
    case 'Azan-Mallory'
        M = [0.853033, 0.508733, 0.112656;
                 0.09289875, 0.8662008, 0.49098468;
                 0.10732849, 0.36765403, 0.9237484];
    case 'Masson Trichrome'
        M = [0.7995107, 0.5913521, 0.10528667;
                0.09997159, 0.73738605, 0.6680326];
    case 'Alcian blue & H'
        M = [0.874622, 0.457711, 0.158256;
                0.552556, 0.7544, 0.353744];
    case 'H PAS'
        M = [0.644211, 0.716556, 0.266844;
                0.175411, 0.972178, 0.154589];
    case 'RGB'
        M = [0, 1, 1;
                1, 0, 1;
                1, 1, 0];
    case 'CMY'
        M = [1, 0, 0;
                0, 1, 0;
                0, 0, 1];
end

if nargin < 1
    I = imread('./Color-Deconvolution/figure9.jpg');
end

% Convert to double
I = double(I);

%% Add third Stain vector, if only two stain vectors are provided. 
% This stain vector is obtained as the cross product of first two
% stain vectors 
if size (M,1) < 3
    M = [M; cross(M(1, :), M(2, :))];
end

% Normalise the input so that each stain vector has a Euclidean norm of 1
M = (M./repmat(sqrt(sum(M.^2, 2)), [1 3]));
%% Sanity check

[h, w, c] = size(I);

% Image must be RGB
if c<3
    error('Image must be RGB'); 
elseif c>3 
    I = I(:,:,1:3);
end

%% MAIN IMPLEMENTATION OF METHOD

% the intensity of light entering the specimen (see section 2a of [1])
Io = 255;

% Vectorize
J = reshape(I, [], 3);

% calculate optical density
OD = -log((J+1)/Io);
Y = reshape(OD, [], 3);

% determine concentrations of the individual stains
% M is 3 x 3,  Y is N x 3, C is N x 3
C = Y / M;
%C = Y * pinv(M);

% Stack back deconvolved channels
DCh = reshape(C, h, w, 3);

end
