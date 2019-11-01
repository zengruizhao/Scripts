function c = parseCamelyon16XmlAnnotations_ccf(xmlPath)
xmlDocument = xmlread(xmlPath);
import javax.xml.xpath.*;
factory = XPathFactory.newInstance();
xpath = factory.newXPath();
% XPath expressions used to parse the XML file
%%
% annotationExpr =xpath.compile('//Annotation');
% PartOfGroup = xpath.compile('//Annotation/@Id');
% annotations = annotationExpr.evaluate(xmlDocument, XPathConstants.NODESET);
% for i_annotations=annotations.getLength():-1:1;
%     a_current = annotations.item(i_annotations-1);
%     k = PartOfGroup.evaluate(a_current, XPathConstants.NODESET);
% end
%%
RegionExpr = xpath.compile('//Region');
PartOfGroup2=xpath.compile('//Region/@Id');
xExpr = xpath.compile('.//Vertex/@X');
yExpr = xpath.compile('.//Vertex/@Y');
regions = RegionExpr.evaluate(xmlDocument, XPathConstants.NODESET);
fprintf(' There are %d area ...\n ',regions.getLength());
for i_regions = regions.getLength():-1:1;
%     fprintf(' %d to be Operated...\n ',i_regions);
    current = regions.item(i_regions-1);
    
    p=PartOfGroup2.evaluate(current, XPathConstants.NODESET);
    x = xExpr.evaluate(current, XPathConstants.NODESET);
    y = yExpr.evaluate(current, XPathConstants.NODESET);
    
    currentCoordinates = zeros(x.getLength(), 2);
    %Part=zeros(p.getLength(), 1);
    Part(1,i_regions)={p.item(i_regions-1).getValue};
    %   currentCoordinates(i_annotations, 2) = ...
    %    % convert the coordinates to a MATLAB array
%     for i_vertex = 1:x.getLength()
%         currentCoordinates(i_vertex, 1) = ...
%             floor(str2double(x.item(i_vertex-1).getValue));
%         currentCoordinates(i_vertex, 2) = ...
%             floor(str2double(y.item(i_vertex-1).getValue));
%     end
   for i_vertex = 1:x.getLength()
        currentCoordinates(i_vertex, 1) = ...
           round(str2double(x.item(i_vertex-1).getValue));
        currentCoordinates(i_vertex, 2) = ...
           round(str2double(y.item(i_vertex-1).getValue));
   end

    coordinates{i_regions,1} = currentCoordinates;
end
Part = Part';
c=[coordinates Part];

end
