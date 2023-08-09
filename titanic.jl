using JLD2, DecisionTree, HTTP, JSON3


model = load_object("titanic.jld2")
passenger = Dict(
    "Pclass" => 1,
    "Sex" => 1,
    "age" => 20,
    "SibSp" => 1,
    "Parch" => 1,
    "Fare" => 100,
    "Embarked" => 2
)

function isSurvived(inputs, model)
    vals = reshape(vals, 1, 7)
    return predict(model, vals)[1]
end

# isSurvived(passenger, model)


function handle(req)
    if req.method == "POST"
        form = JSON3.read(String(req.body))
        # println(typeof(form))
        # println(form[0])
        # passenger = Dict(
        #     "Pclass" => form.pclass,
        #     "Sex" => form.sex.
        #     "age" => form.age,
        #     "SibSp" => form.sibsp,
        #     "Parch" => form.parch,
        #     "Fare" => form.fare,
        #     "Embarked" => form.embarked
        # )

        survived = Float64.(values(form))
        survived = reshape(survived, 1, 7)
        survived = predict(model, survived)[1]
        println(survived)
        # headers = Dict("Access-Control-Allow-Origin" => "http://127.0.0.1:8080")

        return HTTP.Response(200,"$survived")
    end
    return HTTP.Response(200,read("form.html"))
end

HTTP.serve(handle, 8080)