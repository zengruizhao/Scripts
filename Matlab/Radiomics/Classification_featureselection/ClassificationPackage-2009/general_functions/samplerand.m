function randsamples = samplerand(samplength,low,high)

%function randsamples = samplerand(samplength,low,high)
%Get a list of samples of length SAMPLENGTH between values LOW & HIGH
%JC, Jun 2008

url = java.net.URL(['http://www.random.org/integers/?num=', num2str(samplength), '&min=', num2str(low) '&max=', num2str(high), '&col=1&base=10&format=plain&rnd=new']);
is = openStream(url);
isr = java.io.InputStreamReader(is);
br = java.io.BufferedReader(isr);
% list=zeros(samplength,1);
for k=1:samplength,
    list(k)=readLine(br);
end

randsamples=str2num(char(list));

end
