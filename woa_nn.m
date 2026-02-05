function [best_cost, best_nn_params, convergence_curve] = woa_nn(X, y, input_layer_size, ...
                                                              hidden_layer_size, num_labels, lambda, woa_params)
% Extract WOA parameters
max_iter = woa_params.max_iter;
pop_size = woa_params.pop_size;
dim = woa_params.dim;

% Initialize the positions of search agents
positions = rand(pop_size, dim) * 2 - 1; % Random values between -1 and 1

% Initialize convergence curve
convergence_curve = zeros(max_iter, 1);

% Calculate fitness for each search agent
fitness = zeros(pop_size, 1);
for i = 1:pop_size
    fitness(i) = nnCostFunction(positions(i,:)', input_layer_size, ...
                              hidden_layer_size, num_labels, X, y, lambda);
end

% Find the best solution
[best_cost, best_idx] = min(fitness);
best_nn_params = positions(best_idx, :);

% Main loop
for iter = 1:max_iter
    % Update a, A, C, l, and p
    a = 2 - iter * (2 / max_iter);  % a decreases linearly from 2 to 0
    a2 = -1 + iter * (-1 / max_iter); % a2 decreases linearly from -1 to -2

    for i = 1:pop_size
        r1 = rand(); % r1 is a random number in [0,1]
        r2 = rand(); % r2 is a random number in [0,1]

        A = 2 * a * r1 - a;
        C = 2 * r2;

        % Parameters for spiral updating position
        l = (a2 - 1) * rand() + 1; % random number in [-1,1]
        p = rand(); % p in [0,1]

        for j = 1:dim
            % Update position
            if p < 0.5
                if abs(A) < 1
                    % Encircling prey
                    D_alpha = abs(C * best_nn_params(j) - positions(i,j));
                    positions(i,j) = best_nn_params(j) - A * D_alpha;
                else
                    % Search for prey
                    rand_idx = randi([1, pop_size]);
                    X_rand = positions(rand_idx, :);
                    D_rand = abs(C * X_rand(j) - positions(i,j));
                    positions(i,j) = X_rand(j) - A * D_rand;
                end
            else
                % Spiral updating position
                D_best = abs(best_nn_params(j) - positions(i,j));
                positions(i,j) = D_best * exp(l) .* cos(l * 2 * pi) + best_nn_params(j);
            end
        end

        % Calculate fitness of new solution
        current_fitness = nnCostFunction(positions(i,:)', input_layer_size, ...
                                       hidden_layer_size, num_labels, X, y, lambda);

        % Update if the new solution is better
        if current_fitness < fitness(i)
            fitness(i) = current_fitness;
        end

        % Update the best solution
        if fitness(i) < best_cost
            best_cost = fitness(i);
            best_nn_params = positions(i, :);
        end
    end

    % Store the best cost in each iteration
    convergence_curve(iter) = best_cost;

    % Display iteration information
    fprintf('Iteration %d, Best Cost = %f\n', iter, best_cost);
end
end
