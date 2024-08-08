const socket = io();

let graphs = {};

socket.on('new_graph', function(data) {
    graphs[data.title] = {values: [], color: "steelblue"};
    updateGraph(data.title)
});

socket.on('new_graph_value', function(data) {
    graphs[data.title].values.push(data.value);
    updateGraph(data.title);
});

socket.on('set_graph_color', function(data) {
    graphs[data.title].color = data.value;
    if (graphs[title].values.length < 2){
        var graph_line = document.getElementById(`graph-${data.title}-svg-line`);
        graph_line.setAttribute("style", "stroke: "+data.value);
    }
});

socket.on('new_bar_value', function(data) {

    updateBar(data);
});

function updateBar(data){
    data.value = Math.max(Math.min(data.value, 1), 0);
    const barsContainer = document.getElementById('bars-board')
    if (!barsContainer.querySelector("[id=\"bar-box-"+data.title+"\"]")){
        var newElement = document.createElement('div');
        newElement.id = 'bar-box-'+data.title;
        newElement.classList.add('bar-box');
        newElement.classList.add('stat-box');
        newElement.innerHTML = `<div class='bar-title'>${decodeURIComponent(data.title)}</div> <div id='bar-${data.title}'></div>`;
        barsContainer.appendChild(newElement);


        var container = document.getElementById(`bar-${data.title}`);
        var backRect = document.createElement("div");
        backRect.setAttribute('class', 'bar-background-rect');
        backRect.setAttribute("id", `bar-background-rect-${data.title}`)

        var progRect = document.createElement("div");
        progRect.setAttribute("id", `bar-progress-rect-${data.title}`)
        progRect.setAttribute('class', 'bar-progress-rect');

        backRect.appendChild(progRect);
        container.appendChild(backRect);
    }
    document.getElementById(`bar-progress-rect-${data.title}`).style.width = data.value*386+"px";
}



function updateGraph(title){
    const trace = {
        x: Array.from({length: graphs[title].values.length}, (_, i) => i + 1),
        y: graphs[title].values,
        type: 'scatter'
    };
    const graphsContainer = document.getElementById('graphs-board')
    if (!graphsContainer.querySelector("[id=\"graph-box-"+title+"\"]")){
        var newElement = document.createElement('div');
        newElement.id = 'graph-box-'+title;
        newElement.classList.add('graph-box');
        newElement.classList.add('stat-box');
        newElement.innerHTML = `<div class='graph-title'>${decodeURIComponent(title)}</div> <div id='graph-${title}'></div>`;
        graphsContainer.appendChild(newElement);
    }
    buildLineChart(title);
    //Plotly.newPlot('graph-'+title, [trace], {}, {"displayModeBar": false});
}

function buildLineChart(title){
    const containerId = `graph-${title}`;
    const xData = Array.from({length: graphs[title].values.length}, (_, i) => i + 1)
    const yData = graphs[title].values;
    const width = 23*16;
    const height = 13*16;
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    const xMin = 1;//Math.min(...xData);
    const xMax = Math.max(...xData);
    const yMin = 0;//Math.min(...yData);
    const yMax = Math.max(...yData);

    const xScale = d => margin.left + (d - xMin) * (width - margin.left - margin.right) / (xMax - xMin);
    const yScale = d => height - margin.bottom - (d - yMin) * (height - margin.top - margin.bottom) / (yMax - yMin);

    var container = document.getElementById(containerId);
    container.innerHTML = `<svg width="25rem" height="15rem" id="${containerId+'-svg'}" class="graph-svg"></svg>`
    const svg = document.getElementById(containerId+'-svg');

    if (graphs[title].values.length >= 2) {

        const lineGenerator = (xData, yData) => {
            return xData.map((d, i) => {
                const x = xScale(d);
                const y = yScale(yData[i]);
                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
            }).join(' ');
        };

        const linePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        linePath.setAttribute('class', 'line');
        linePath.setAttribute('id', `${containerId}-svg-line`);
        linePath.setAttribute('d', lineGenerator(xData, yData));
        linePath.setAttribute("style", "stroke: "+graphs[title].color);

        svg.appendChild(linePath)
    }

    if (graphs[title].values.length >= 1) {
        for (var i=0; i<graphs[title].values.length; i++){
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', xScale(i+1));
            circle.setAttribute('cy', yScale(graphs[title].values[i]));
            circle.setAttribute('r', 3);
            circle.setAttribute('fill', 'white');
            circle.setAttribute('stroke', graphs[title].color);
            circle.setAttribute('stroke-width', 1);
            svg.appendChild(circle);
        }
    }

}
