const socket = io();

let values = {};

socket.on('new_graph_value', function(data) {

    if (data.title in values){
        values[data.title].push(data.value);
    }else{
        values[data.title] = [data.value];
    }
    updateGraph(data.title);
});

socket.on('new_bar_value', function(data) {
    updateBar(data);
});

function updateBar(data){

    const barsContainer = document.getElementById('bars-board')
    if (!barsContainer.querySelector("[id=\"bar-box-"+data.title+"\"]")){
        var newElement = document.createElement('div');
        newElement.id = 'bar-box-'+data.title;
        newElement.classList.add('bar-box');
        newElement.innerHTML = `<div class='bar-title'>${decodeURIComponent(data.title)}</div> <div id='bar-${data.title}'></div>`;
        barsContainer.appendChild(newElement);
    }
    var containerId = `bar-${data.title}`;
    var container = document.getElementById(containerId);
    const backRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    backRect.setAttribute("x", "0");
    backRect.setAttribute("y", "0");
    backRect.setAttribute("width", "100");
    backRect.setAttribute("height", "10");
    backRect.setAttribute("fill", "#000000");

    const progRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    progRect.setAttribute("x", "0");
    progRect.setAttribute("y", "0");
    progRect.setAttribute("width", data.value);
    progRect.setAttribute("height", "10");
    progRect.setAttribute("fill", "#ff0000");

    container.innerHTML = `<svg width="25rem" height="8rem" id="${containerId+'-svg'}" class="bar-svg"></svg>`
    document.getElementById(containerId+'-svg').appendChild(backRect)
    document.getElementById(containerId+'-svg').appendChild(progRect)
}


function updateGraph(title){
    const trace = {
        x: Array.from({length: values[title].length}, (_, i) => i + 1),
        y: values[title],
        type: 'scatter'
    };
    const graphsContainer = document.getElementById('graphs-board')
    if (!graphsContainer.querySelector("[id=\"graph-box-"+title+"\"]")){
        console.log(title)
        var newElement = document.createElement('div');
        newElement.id = 'graph-box-'+title;
        newElement.classList.add('graph-box');
        newElement.innerHTML = `<div class='graph-title'>${decodeURIComponent(title)}</div> <div id='graph-${title}'></div>`;
        graphsContainer.appendChild(newElement);
    }
    buildLineChart(`graph-${title}`, Array.from({length: values[title].length}, (_, i) => i + 1), values[title]);
    //Plotly.newPlot('graph-'+title, [trace], {}, {"displayModeBar": false});
}

function buildLineChart(containerId, xData, yData){
    const width = 23*16;
    const height = 13*16;
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    const yMin = Math.min(...yData);
    const yMax = Math.max(...yData);

    const xScale = d => margin.left + (d - xMin) * (width - margin.left - margin.right) / (xMax - xMin);
    const yScale = d => height - margin.bottom - (d - yMin) * (height - margin.top - margin.bottom) / (yMax - yMin);

    const lineGenerator = (xData, yData) => {
        return xData.map((d, i) => {
            const x = xScale(d);
            const y = yScale(yData[i]);
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        }).join(' ');
    };

    var container = document.getElementById(containerId);
    const linePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    linePath.setAttribute('class', 'line');
    linePath.setAttribute('d', lineGenerator(xData, yData));
    container.innerHTML = `<svg width="25rem" height="15rem" id="${containerId+'-svg'}" class="graph-svg"></svg>`
    document.getElementById(containerId+'-svg').appendChild(linePath)
}
