%% Initialization
%  Initialize the world, Q-table, and hyperparameters


learning_rate = 0.2; 
discount_factor = 0.95; 
EPISODES = 1000; 
world_nr = 10;

world = gwinit(world_nr);
Q_table = rand(world.ysize, world.xsize, 4);

state = gwstate();

%Top
Q_table(1,:,2) = -Inf;
%Left
Q_table(:,1,4) = -Inf;
%Bottom
Q_table(size(Q_table, 1),:,1) = -Inf;
%Right
Q_table(:,size(Q_table, 2),3) = -Inf;

%% Training loop
%  Train the agent using the Q-learning algorithm.
for i = 1:EPISODES
    state = gwstate();
    gwinit(world_nr);
    while(1==1)
        [action, opt_action] = chooseaction(Q_table, state.pos(1), state.pos(2), [1, 2, 3, 4],[0.25, 0.25, 0.25, 0.25], 1);
        state_next = gwaction(action);
        reward = state_next.feedback;
        q = getvalue(Q_table);
        if (state_next.isterminal)
            break;
        else
            Q_table(state.pos(1), state.pos(2), action) =  Q_table(state.pos(1),state.pos(2),action) + learning_rate * (reward + discount_factor * q(state_next.pos(1), state_next.pos(2), 1) - Q_table(state.pos(1), state.pos(2), action));
        %%else
        %%    Q_table(state.pos(1), state.pos(2), action) = -inf;
        end
        state = state_next;
    end
end


%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traver
% ridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.
P = getpolicy(Q_table);
gwinit(world_nr);
state = gwstate();
while(state.isterminal ~= 1)
    optimal_action = P(state.pos(1),state.pos(2));
    %%gwplotarrow(state.pos, optimal_action);
    state = gwaction(optimal_action);
    gwdraw(EPISODES, P);
end
getvalue(Q_table)
