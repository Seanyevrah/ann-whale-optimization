clear;
close all;
clc;

% Load and prepare data
data = load('data_banknote_authentication.txt');
X = data(:, 1:4);
y = data(:, 5) + 1; % Convert labels to 1 and 2 (since Octave indexes from 1)

% Parameters
input_layer_size = 4;  % 4 features
hidden_layer_size = 5; % 5 hidden units
num_labels = 2;        % 2 classes (0 and 1)
lambda = 0.1;          % Regularization parameter

% Split data into training and testing sets
[m, ~] = size(X);
train_ratio = 0.7;
train_size = floor(train_ratio * m);
X_train = X(1:train_size, :);
y_train = y(1:train_size, :);
X_test = X(train_size+1:end, :);
y_test = y(train_size+1:end, :);

% Normalize features
[X_train, mu, sigma] = featureNormalize(X_train);
X_test = normalizeUsing(X_test, mu, sigma);

% Initialize WOA parameters
woa_params.max_iter = 100;    % Maximum number of iterations
woa_params.pop_size = 100;    % Population size
woa_params.dim = (input_layer_size + 1) * hidden_layer_size + ...
                 (hidden_layer_size + 1) * num_labels; % Dimension of the problem

% Run WOA to optimize neural network weights
[best_cost, best_nn_params, woa_costs] = woa_nn(X_train, y_train, input_layer_size, ...
                                                hidden_layer_size, num_labels, lambda, woa_params);

% Train neural network with optimized weights
Theta1 = reshape(best_nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(best_nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Predict on training set
pred_train = predict(Theta1, Theta2, X_train);
train_accuracy = mean(double(pred_train == y_train)) * 100;

% Predict on test set
pred_test = predict(Theta1, Theta2, X_test);
test_accuracy = mean(double(pred_test == y_test)) * 100;

fprintf('\nTraining Set Accuracy: %f\n', train_accuracy);
fprintf('Test Set Accuracy: %f\n', test_accuracy);
fprintf('Final Cost: %.6f\n', best_cost);

% Plot convergence
figure;
plot(1:woa_params.max_iter, woa_costs, 'LineWidth', 2);
title('WOA Convergence');
xlabel('Iteration');
ylabel('Best Cost');
grid on;

