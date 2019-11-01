function annotation = getAnnotation(xmlFile)

% debugging % 
% xmlFile = '8863.xml';
%%%%%%%%%%%%%

xDoc = xmlread(xmlFile);
theStruct = parseChildNodes(xDoc);
Regions = xDoc.getElementsByTagName('Region');

annotation = [];
for i=0:Regions.getLength-1
    this_Region = Regions.item(i);
    
    % grab ROI ID and area
    ROI_id      = cell(this_Region.getAttribute('Id'));      
    ROI_id      = str2double(ROI_id{1});
    ROI_area    = cell(this_Region.getAttribute('Area'));  
    ROI_area    = str2double(ROI_area{1});
    ROI_type    = cell(this_Region.getAttribute('Type'));  
    ROI_type    = str2double(ROI_type{1});
    ROI_text    = cell(this_Region.getAttribute('Text'));  
    
    % find child containing vertices for current ROI
    Vertices = this_Region.getFirstChild;
    while ~strcmpi(Vertices.getNodeName,'Vertices')
        Vertices = Vertices.getNextSibling;
    end
    
    % get vertices for current ROI
    X=[]; Y=[];
    Vertex = Vertices.getFirstChild;
    while ~isempty(Vertex)
        if strcmpi(Vertex.getNodeName,'Vertex')
            x = cell(Vertex.getAttribute('X')); x = str2double(x{1});
            y = cell(Vertex.getAttribute('Y')); y = str2double(y{1});
            X = [X; x];
            Y = [Y; y];
        end
        
        Vertex = Vertex.getNextSibling;
    end
    
    annotation(i+1).roi_id  = ROI_id;
    annotation(i+1).area    = ROI_area;
    annotation(i+1).type    = ROI_type;
    annotation(i+1).text    = ROI_text;
    annotation(i+1).X       = X;
    annotation(i+1).Y       = Y;
end
