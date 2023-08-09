using DecisionTree
using Statistics
using Random
using DataFrames
using JET
using CSV
using Plots
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.GridSearch: GridSearchCV
using JLD2

## set of classification parameters and respective default values
# n_subfeatures: #*number of features to consider at random per split (default: -1, sqrt(# features))
n_subfeatures = -1
# n_trees: #*number of trees to train (default: 10)
n_trees = 50
# partial_sampling: #* fraction of samples to train each tree on (default: 0.7)
partial_sampling = 0.7
# max_depth: #* maximum depth of the decision trees (default: no maximum)
max_depth = -1
# min_samples_leaf: #* the minimum number of samples each leaf needs to have (default: 5)
min_samples_leaf = 12
# min_samples_split: #* the minimum number of samples in needed for a split (default: 2)
min_samples_leaf = 7
# min_purity_increase: #* minimum purity needed for a split (default: 0.0)
min_samples_split = 3
min_purity_increase = 0.0
# keyword rng: #* the random number generator or seed to use (default Random.GLOBAL_RNG)
seed = 3
## multi-threaded forests must be seeded with an `Int`


#* preprocessing function
function preprocess(df)
    #* dropping missing embarked data
    df = dropmissing(df, :Embarked)

    #* handling missing age data - filling with median
    median_age = median(skipmissing(df[:, :Age]))
    # println("the median age : ", median_age)
    df.Age = replace(df.Age, missing => median_age)

    #* Dropping missing data
    df = select(df, Not(:Cabin, :PassengerId, :Name, :Ticket))

    #* handling catergorical data
    df.Embarked = Int64.(replace(df.Embarked, "S" => 1, "C" => 2, "Q" => 3))
    df.Sex = Int64.(replace(df.Sex, "male" => 1, "female" => 2))

    #* handling Targets
    df.Survived = String.(replace(df.Survived, 0 => "Died", 1 => "Survived"))
    return df
end

#* builing the random forest 
function buling_rf(features, labels)
    model = build_forest(labels, features,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase;
        rng=seed)
    return model
end

#* calculate the accuracy of the model
function calc_acc(model, features, labels)
    results = (apply_forest(model, features) .== labels)
    acc = sum(results)/length(results)
    return acc
end


#* adjusting Train df 
train_df = CSV.read("train.csv", DataFrame)
train_df = preprocess(train_df)


#* adjusting Test df
test_df = CSV.read("test.csv", DataFrame)
test_targets = CSV.read("gender_submission.csv", DataFrame)
test_df = innerjoin(test_df, test_targets, on=:PassengerId)
test_df = preprocess(test_df)
test_df = dropmissing(test_df, :Fare)


#* Building the model
train_features = Float64.(Matrix(train_df[:, Not(:Survived)]))
train_labels = train_df[:, :Survived]

test_features = Float64.(Matrix(test_df[:, Not(:Survived)]))
test_labels = test_df[:, :Survived]


model = buling_rf(train_features, train_labels)
train_acc = calc_acc(model, train_features, train_labels)
test_acc = calc_acc(model, test_features, test_labels)

println("train accuracy :", train_acc)
println("test accuracy :", test_acc)
apply_forest(model, test_features)


using JLD2
save_object("titanic.jld2", model)

passenger = Dict(
    "Pclass" => 3,
    "Sex" => 1,
    "age" => 34.5,
    "SibSp" => 0,
    "Parch" => 0,
    "Fare" => 7.8292,
    "Embarked" => 3
)

function isSurvived(inputs, model)
    vals = reshape(vals, 1, 7)
    return predict(model, vals)[1]
end


model = load_object("titanic.jld2")
passenger = Float64.(values(passenger))
passenger = reshape(passenger, 1, 7)
survived = apply_forest(model, passenger)