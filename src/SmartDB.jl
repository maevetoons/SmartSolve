export compute_feature_values

all_features = OrderedDict()
all_features[:length] = x -> length(x)
all_features[:n_rows] = x -> size(x, 1)
all_features[:n_cols] = x -> size(x, 2)
all_features[:rank] = x -> rank(x)
all_features[:condnumber] = x -> cond(Array(x), 2)
all_features[:sparsity] = x -> count(iszero, x) / length(x)
all_features[:isdiag] = x -> Float64(isdiag(x))
all_features[:issymmetric] = x -> Float64(issymmetric(x))
all_features[:ishermitian] = x -> Float64(ishermitian(x))
all_features[:isposdef] = x -> Float64(isposdef(x))
all_features[:istriu] = x -> Float64(istriu(x))
all_features[:istril] = x -> Float64(istril(x))

function compute_feature_dict(A; features = keys(all_features))
    feature_dict = OrderedDict()
    for f in features
        feature_dict[f] = all_features[f](A)
    end
    return feature_dict
end

function compute_feature_values(A; features = keys(all_features))
    feature_vals = Float64[]
    for f in features
        push!(feature_vals, all_features[f](A))
    end
    return feature_vals
end

function create_empty_db()
    df1 = DataFrame(n_experiment = Int[],
                    pattern = String[])
    features = compute_feature_dict(rand(3,3))
    column_names = keys(features)
    column_types = map(typeof, values(features))
    df2 = DataFrame(OrderedDict(k => T[] for (k, T) in zip(column_names, column_types)))
    df3 = DataFrame(algorithm = String[],
                    time = Float64[],
                    error = Float64[])
    return hcat(df1, df2, df3)
end

function get_smart_choices(db, mat_patterns, ns)
    db_opt = create_empty_db()
    for mat_pattern in mat_patterns
        for n in ns
            db′ = @views db[(db.pattern .== mat_pattern) .&&
                            (db.n_cols .== n), :]
            if length(db′.time) > 0
                min_time = minimum(db′.time)
                min_time_row = db′[db′.time .== min_time, :][1, :]
                push!(db_opt, min_time_row)
            end
        end
    end
    return db_opt
end

#------------------------------------------------------------------------------
function get_best_feature_combos(rankedfeatures)
#Run algorithm using greedy forward approach to find best combination
#Parameters: error, time
#------------------------------------------------------------------------------

    #NOT GOOD
    "loop curr_error > error_tol and cur_time > time_tol

        push!(curr_features, features[i])" # i is the iteration number

        # Fit the ML model with curr_features

        "curr_error = measure ML error, use accuracy"
    x = A \ b;
    norm(A * x - b, 1)
    x = lu(A) \ b;
    norm(A * x - b, 1)
    x = smartlu(A) \ b;
    norm(A * x - b, 1)
        "curr_time = measure ML time, use @benchmark"
    @benchmark

#Output: List of top (5) best combinations
    return best feature combos
end

#------------------------------------------------------------------------------
function smartfeatures(db)
#Computes the contribution of each feature to reduce the error using Shapley values.
#------------------------------------------------------------------------------
#1. Calculate Shapley values.

    #isolate features and error
    error,features = unpack(db[:, Cols(Between(:"length",:"istril"), :"error")], ==(:error))
    #idea: add split parallel functionality for time and error

    #instantiate ML model
    decision_tree = @load DecisionTreeRegressor pkg = "DecisionTree"
    model = machine(decision_tree, features, error)
    fit!(model)

    #wrapper function
    function predict_function(model, data)
        data_pred = DataFrame(y_pred = predict(model, data))
        return data_pred
    end

    #compute stochastic Shapley values.
    explain = copy(db)
    explain = select(explain, Not(Symbol(outcome_name)))

    shapley = ShapML.shap(explain = explain,
                        model = model,
                        predict_function = predict_function,
                        sample_size = 60
                        )

    show(shapley, allcols = true)
#------------------------------------------------------------------------------
#2. computes the computational cost of each feature.
    "Take into account variable interpolation and number of samples (>100)."

    #sort features by score average importance/time.
    shapley = hcat(shapley, time=repeat(db[!,:time], inner=sum(unique(shapley[!,:feature_name]))))

    #ERROR: nested task error: type SubArray has no field shap_effect
    bestfeatures = combine(groupby(shapley, :feature_name),
                            [:shap_effect, :time] => (x,y) -> mean_effect = mean(abs(x.shap_effect)/(y.time)))

    bestfeatures = sort(bestfeatures, order(:mean_effect, rev = true))

    #show plot (optional)
    baseline = round(shapley.intercept[1], digits = 1)
    p = plot(bestfeatures, y = :feature_name, x = :mean_effect, Coord.cartesian(yflip = true),
             Scale.y_discrete, Geom.bar(position = :dodge, orientation = :horizontal),
             Theme(bar_spacing = 1mm),
             Guide.xlabel("Mean Absolute Shapley Value/Time (baseline = $baseline)"), Guide.ylabel(nothing),
             Guide.title("Feature Importance"))

    return get_best_feature_combos(bestfeatures[!,:feature_name])
end
#------------------------------------------------------------------------------





#Misc:
    function compute_score(A) #find better formatting
        for A in samples
            A = matrixdepot("poisson", round(Int, sqrt(n)));

            lu_res = @benchmark lu($A)
            smartlu_res = @benchmark smartlu($A)
            b = rand(n);
            bs_res = @benchmark $A\$b
            lu_bs_res = @benchmark lu($A)\$b
            smartlu_bs_res = @benchmark smartlu($A)\$b
            return median(lu_res.times), median(smartlu_res.times), median(bs_res.times), median(lu_bs_res.times), median(smartlu_bs_res.times)
        end
    end
