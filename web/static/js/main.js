 

 // Columns to exclude
 var excludeColumns = ['Centroid Membership', 'TSNE1', 'TSNE2'];

 // Generate table headers
 var thead = document.querySelector('#clustering-result-table thead');
 var headerRow = document.createElement('tr');
 var tableHeaders = Object.keys(tableData[0]).filter(function(header) {
     return !excludeColumns.includes(header);
 });

 tableHeaders.forEach(function(header) {
     var th = document.createElement('th');
     th.innerText = header;
     headerRow.appendChild(th);
 });
 thead.appendChild(headerRow);

 // Insert table data
 var tbody = document.querySelector('#clustering-result-table tbody');
 tableData.forEach(function(rowData) {
     var row = document.createElement('tr');
     tableHeaders.forEach(function(header) {
         var cell = document.createElement('td');
         var cellData = rowData[header];
         if (Array.isArray(cellData)) {
             cell.innerText = cellData.join(', ');
         } else {
             cell.innerText = cellData;
         }
         row.appendChild(cell);
     });
     tbody.appendChild(row);
 });