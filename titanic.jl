using JLD2, DecisionTree, HTTP, JSON3


model = load_object("titanic.jld2")


function isSurvived(inputs, model)
    vals = reshape(vals, 1, 7)
    return predict(model, vals)[1]
end



function handle(req)
    if req.method == "POST"
        form = JSON3.read(String(req.body))
        survived = Float64.(values(form))
        survived = reshape(survived, 1, 7)
        survived = apply_forest(model, survived)[1]
        
        println(survived)
        if survived .== "Survived"
            survived = 1
        else 
            survived = 0
        end
        # headers = Dict("Access-Control-Allow-Origin" => "http://127.0.0.1:8080")
        return HTTP.Response(200,"$survived")
    end
    return HTTP.Response(200,read("form.html"))
end

HTTP.serve(handle, 8080)