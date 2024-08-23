const socket = io();

let runData = {graphs: {}, bars: {}};
let graphs = runData.graphs;
let bars = runData.bars;

let lmThinking = false;

socket.emit("fill_me_in");

socket.on('full_data_drop', function(data) {
    runData = data;
    graphs = runData.graphs;
    bars = runData.bars;
    console.log(runData);
    const graphNames = Object.keys(graphs)
    for (let i=0; i<graphNames.length; i++){
        updateGraph(graphNames[i]);
    }

    const barNames = Object.keys(bars)
    for (let i=0; i<barNames.length; i++){
        updateBar(barNames[i]);
    }

    for (let i=0; i<data.chat.length; i++){
        if (data.chat[i].sender == 0) {
            addLmMessage(data.chat[i].message);
        } else {
            addUserMessage(data.chat[i].message);
        }
    }
})

socket.on('new_graph', function(data) {
    graphs[data.title] = {values: [], color: "steelblue"};
    updateGraph(data.title)
});

socket.on('new_bar', function(data) {
    bars[data.title] = {value: 0, color: "steelblue"};
    updateBar(data.title)
});

socket.on('new_graph_value', function(data) {
    console.log(graphs);
    graphs[data.title].values.push(data.value);

    updateGraph(data.title);
});

socket.on('set_graph_color', function(data) {
    graphs[data.title].color = data.value;
    if (graphs[data.title].values.length >= 2){
        var graph_line = document.getElementById(`graph-${data.title}-svg-line`);
        graph_line.setAttribute("style", "stroke: "+data.value);
    }
});

socket.on('new_bar_value', function(data) {
    bars[data.title].value = data.value;
    updateBar(data.title);
});


socket.on('set_bar_color', function(data) {
    bars[data.title].color = data.value;
    var bar_fill = document.getElementById(`bar-progress-rect-${data.title}`);
    bar_fill.setAttribute("style", "background: "+data.value);

});

socket.on('lm_message', function(data){
    addLmMessage(data.message);
    lmThinking = false;
    document.getElementById("chat-input").disabled = false;
})

function updateBar(title){
    data = bars[title];
    data.value = Math.max(Math.min(data.value, 1), 0);
    const barsContainer = document.getElementById('bars-board')
    if (!barsContainer.querySelector("[id=\"bar-box-"+title+"\"]")){
        var newElement = document.createElement('div');
        newElement.id = 'bar-box-'+title;
        newElement.classList.add('bar-box');
        newElement.classList.add('stat-box');
        newElement.innerHTML = `<span class='bar-title'>${decodeURIComponent(title)}</span> <span class='bar-number' id='bar-${title}-number'></span> <div id='bar-${title}'></div>`;
        barsContainer.appendChild(newElement);

        var container = document.getElementById(`bar-${title}`);
        var backRect = document.createElement("div");
        backRect.setAttribute('class', 'bar-background-rect');
        backRect.setAttribute("id", `bar-background-rect-${title}`)

        var progRect = document.createElement("div");
        progRect.setAttribute("id", `bar-progress-rect-${title}`)
        progRect.setAttribute('class', 'bar-progress-rect');
        progRect.setAttribute("style", "background: "+data.color);

        backRect.appendChild(progRect);
        container.appendChild(backRect);
    }
    document.getElementById(`bar-progress-rect-${title}`).style.width = data.value*386 + "px";
    document.getElementById(`bar-${title}-number`).innerHTML = (data.value*100).toFixed(2) + "%";

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
    const padding = { top: 10, right: 10, bottom: 30, left: 30 };

    const xMin = 1;//Math.min(...xData);
    const xMax = Math.max(...xData);
    const yMin = 0;//Math.min(...yData);
    var yMax = Math.max(...yData);

    const yMarks = tickMarks(yMax);
    yMax = yMarks[yMarks.length-1];

    const xScale = d => padding.left + (d - xMin) * (width - padding.left - padding.right) / (xMax - xMin);
    const yScale = d => height - padding.bottom - (d - yMin) * (height - padding.top - padding.bottom) / (yMax - yMin);

    var container = document.getElementById(containerId);
    container.innerHTML = `<svg width="25rem" height="15rem" id="${containerId+'-svg'}" class="graph-svg"></svg>`
    const svg = document.getElementById(containerId+'-svg');
    const bounding = svg.getBoundingClientRect();

    const yAxisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxisLine.setAttribute("x1", padding.left);
    yAxisLine.setAttribute("y1", padding.top);
    yAxisLine.setAttribute("x2", padding.left);
    yAxisLine.setAttribute("y2", bounding.height-padding.bottom);
    yAxisLine.setAttribute("stroke", "#999");
    svg.appendChild(yAxisLine);

    for (let i=0; i<yMarks.length; i++){
        const markNum = yMarks[i];
        const mark = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        mark.setAttribute("y1", yScale(markNum));
        mark.setAttribute("y2", yScale(markNum));
        mark.setAttribute("x1", padding.left-3);
        mark.setAttribute("x2", padding.left);
        mark.setAttribute("stroke", "#999");
        svg.appendChild(mark);

        const displayNum = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        displayNum.textContent = Math.round(markNum*100)/100;
        displayNum.setAttribute('y', yScale(markNum));
        displayNum.setAttribute('x', padding.left-16);
        displayNum.setAttribute('font-family', "Arial");
        displayNum.setAttribute('font-size', 12);
        displayNum.setAttribute('text-align', 'center');
        displayNum.setAttribute('fill', 'black');
        svg.appendChild(displayNum);
        displayNum.setAttribute('y', yScale(markNum)+displayNum.getBBox().height/2);
        displayNum.setAttribute('x', padding.left-displayNum.getBBox().width-8);
    }

    const xAxisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxisLine.setAttribute("x1", padding.left);
    xAxisLine.setAttribute("y1", bounding.height-padding.bottom);
    xAxisLine.setAttribute("x2", bounding.width-padding.right);
    xAxisLine.setAttribute("y2", bounding.height-padding.bottom);
    xAxisLine.setAttribute("stroke", "#999");
    svg.appendChild(xAxisLine);

    for (let i=1; i<=xData.length; i++){
        const markNum = i;
        let markX = xScale(markNum);
        if (xData.length == 1){
            markX = bounding.width/2;
        }
        const mark = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        mark.setAttribute("x1", markX);
        mark.setAttribute("x2", markX);
        mark.setAttribute("y1", bounding.height-padding.bottom+3);
        mark.setAttribute("y2", bounding.height-padding.bottom);
        mark.setAttribute("stroke", "#999");
        svg.appendChild(mark);

        const displayNum = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        displayNum.textContent = markNum;
        displayNum.setAttribute('x', markX);
        displayNum.setAttribute('y', bounding.height-padding.bottom+16);
        displayNum.setAttribute('font-family', "Arial");
        displayNum.setAttribute('font-size', 12);
        displayNum.setAttribute('text-align', 'center');
        displayNum.setAttribute('fill', 'black');
        svg.appendChild(displayNum);
        displayNum.setAttribute('x', markX-displayNum.getBBox().width/2);
    }

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

        for (var i=0; i<graphs[title].values.length; i++){
            svg.appendChild(svgCircle(xScale(i+1), yScale(graphs[title].values[i]), graphs[title].color));
        }
    }

    if (graphs[title].values.length == 1) {
        svg.appendChild(svgCircle(bounding.width/2, bounding.height/2, graphs[title].color));
    }
}

function svgCircle(x, y, color){
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', x);
    circle.setAttribute('cy', y);
    circle.setAttribute('r', 3);
    circle.setAttribute('fill', 'white');
    circle.setAttribute('stroke', color);
    circle.setAttribute('stroke-width', 1);
    return circle;
}

function sigFigCeil(num, sigFigs){
    if (num == 0) return 0;
    const sign = num > 0 ? 1 : -1;
    let factor = Math.pow(10, sigFigs-Math.ceil(Math.log10(Math.abs(num))));
    return sign * Math.ceil(num * factor) / factor;
}

function tickMarks(max){
    const unit = sigFigCeil(max/10, 2);
    var marks = []
    for (let i=0; i<max+unit; i+=unit){
        marks.push(i);
    }
    return marks;
}

document.getElementById("chat-input").addEventListener("keydown", function(event) {
    if (event.key == "Enter") {
        event.preventDefault();
        addUserMessage(this.value);
        lmThinking = true;
        document.getElementById("chat-input").disabled = true;
        socket.emit("user_message", {message: this.value});
        this.value = "";
    }
})

function addLmMessage(string) {
    const chatBox = document.getElementById("chat-box");
    const newChatMessage = document.createElement("div");
    newChatMessage.classList.add("chat-message","chat-message-lm");
    newChatMessage.textContent = string;
    chatBox.appendChild(newChatMessage);
}


function addUserMessage(string) {
    const chatBox = document.getElementById("chat-box");
    const newChatMessage = document.createElement("div");
    newChatMessage.classList.add("chat-message","chat-message-user");
    newChatMessage.textContent = string;
    chatBox.appendChild(newChatMessage);
}
