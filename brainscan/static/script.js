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

function updateGraph(title) {
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
        newElement.innerHTML = "<h2>"+decodeURIComponent(title)+"</h2> <div id='graph-"+title+"'></div>";
        graphsContainer.appendChild(newElement);
    }

    Plotly.newPlot('graph-'+title, [trace]);
}