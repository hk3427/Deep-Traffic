
//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 50;
patchesAhead = 10;
patchesBehind = 5;
trainIterations = 100000;

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 0;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
layer_defs.push({
    type: 'fc',
    num_neurons: 24,
    activation: 'tanh'
});
layer_defs.push({
    type: 'fc',
    num_neurons: 24,
    activation: 'tanh'
});
layer_defs.push({
    type: 'fc',
    num_neurons: 24,
    activation: 'tanh'
});
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 128,
    l2_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 50000;
opt.start_learn_threshold = 5000;
opt.gamma = 0.9;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
    brain.backward(lastReward);
    var action = brain.forward(state);

    draw_net();
    draw_stats();

    return action;
}

//]]>
    
/*###########*/
if (brain) {
brain.value_net.fromJSON({"layers":[{"out_depth":19,"out_sx":1,"out_sy":1,"layer_type":"input"},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":19,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":19,"w":{"0":-0.3685098424958753,"1":0.24437021656710595,"2":-0.22893010225616867,"3":-0.19078344499637984,"4":-0.37613423184568046,"5":0.15966345065040422,"6":0.1244204179528099,"7":-0.1677988925322477,"8":-0.10354375741693739,"9":0.12085960080573922,"10":0.04815673601472686,"11":0.03633343863244772,"12":-0.25282614314982627,"13":0.16844784352960107,"14":-0.057741578722317326,"15":0.07522144542892896,"16":-0.11351178810223744,"17":0.37025023581117394,"18":-0.22472608135157876}}],"biases":{"sx":1,"sy":1,"depth":1,"w":{"0":0.07535838191833041}}},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"relu"},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":1,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":1,"w":{"0":-2.0843326195592775}},{"sx":1,"sy":1,"depth":1,"w":{"0":0.5146626679650067}},{"sx":1,"sy":1,"depth":1,"w":{"0":1.1466454401954942}},{"sx":1,"sy":1,"depth":1,"w":{"0":0.9378474755998956}},{"sx":1,"sy":1,"depth":1,"w":{"0":-0.3987919319428048}}],"biases":{"sx":1,"sy":1,"depth":5,"w":{"0":5.320811270458415,"1":8.326151063907487,"2":4.648978086284402,"3":4.759152823921189,"4":5.21167423189301}}},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"regression","num_inputs":5}]});
}