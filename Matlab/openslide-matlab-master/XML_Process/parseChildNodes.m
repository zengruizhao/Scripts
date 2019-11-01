function children = parseChildNodes(theNode)
% Recurse over node children.
children = [];
if theNode.hasChildNodes
   childNodes = theNode.getChildNodes;
   numChildNodes = childNodes.getLength;
   allocCell = cell(1, numChildNodes);

   children = struct(             ...
      'Name', allocCell, 'Attributes', allocCell,    ...
      'Data', allocCell, 'Children', allocCell);
    temp = 0;
    for count = 1:numChildNodes
        theChild = childNodes.item(count-1);
        %%
%         if java.lang.String('#text')~=theChild.getNodeName
%             temp = temp + 1;
%             children(temp) = makeStructFromNode(theChild);
%         end
        %%
        children(count) = makeStructFromNode(theChild);
    end
end