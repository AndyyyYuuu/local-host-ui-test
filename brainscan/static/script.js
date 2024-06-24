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
    console.log(values);
    const trace = {
        x: Array.from({length: values[title].length}, (_, i) => i + 1),
        y: values.loss,
        type: 'scatter'
    };

    Plotly.newPlot('graph', [trace]);
}