using Distributed
using Statistics
using LinearAlgebra
using PDMats
using SpecialFunctions
using FFTW
using DSP
using FileIO
using Serialization
#using SciPy
#using GaussianProcesses

#addprocs(8)

#For the HDP, vHDP version
#@everywhere using VersatileHDPMixtureModels
using VersatileHDPMixtureModels


# For the DP version
#@everywhere using DPMMSubClusters



function Initialize_lists(X)
    g_list = []
    μ_idx_list = []
    for i = 1:size(X,1)
        g = zeros(size(X[i]))
        μ_idx = zeros(size(X[i]))
        push!(g_list, g)
        push!(μ_idx_list, μ_idx)
    end
    println("Done Initialize Lists")
    return g_list, μ_idx_list
end


function create_input_rgb(X)
    color_size, x_size, y_size = size(X)
    input_arr = zeros(3,x_size*y_size)
    for i = 1:x_size
        for j = 1:y_size
            input_arr[:, (i - 1) * y_size + j] = [X[1, i, j], X[2, i, j], X[3, i, j]]
        end
    end
    return input_arr
end


function convert_vec_to_rgb(X, g)
    color_size, x_size, y_size = size(X)
    g_reshape = zeros(color_size, x_size, y_size)
    one_color_size = x_size * y_size
    for i in 0:color_size-1
        one_color = g[(one_color_size * i) + 1 : (one_color_size * (i+1))]
        g_reshape[i+1,:,:] = reshape(one_color, (x_size, y_size))
    end
    return g_reshape
end


function create_input_rgb_xy(X)
    color_size, x_size, y_size = size(X)
    input_arr = zeros(5,x_size*y_size)
    for i = 1:x_size
        for j = 1:y_size
            input_arr[:, (i - 1) * y_size + j] = [X[1, i, j], X[2, i, j], X[3, i, j], i, j]
        end
    end
    return input_arr
end

function create_input_rgb_xy_mask(X, mask)
    color_size, x_size, y_size = size(X)
    input_arr = zeros(5,sum(x->x>0, mask))
    idx = 1
    for i = 1:x_size
        for j = 1:y_size
            if mask[i,j] > 0
                #input_arr[:, (i - 1) * y_size + j] = [X[1, i, j], X[2, i, j], X[3, i, j], i, j]
                input_arr[:, idx] = [X[1, i, j], X[2, i, j], X[3, i, j], i, j]
                idx = idx + 1
            end
        end
    end
    return input_arr
end

# ldim is 0 in the hdp part
function create_priors_hdp(gdim, ldim, images_dict, scalar_g, scalar_l, ν_g, v_l, λ_g, λ_l)
    #data_cov = cov(images_dict[1]')
    images_color = hcat(images_dict[1][1:3,:])   
    images_xy = hcat(images_dict[1][4:5,:])  

    mean_data_rgb = mean(images_color,dims = 2)[:]
    mean_data_xy = mean(images_xy,dims = 2)[:]

    cov_data_rgb = cov(images_color')
    cov_data_xy = cov(images_xy')

    g_prior = niw_hyperparams(λ_g, mean_data_rgb, ν_g, cov_data_rgb * scalar_g)
    l_prior = niw_hyperparams(λ_l, mean_data_xy, v_l, cov_data_xy * scalar_l)

    return g_prior, l_prior
end



function create_images_input(X, g, gdim)

    pts = Dict{Any,Any}()
    for i = 1:size(X,1)
        X_minus_g = X[i]-g[i]
        #input_arr = create_input_rgb(X_minus_g)
        input_arr = create_input_rgb_xy(X_minus_g)
        push!(pts, i => input_arr)
    end
    return pts
end


function create_images_mean_input(X, g, gdim, mask, flag_mean_DP)
    pts = Dict{Any,Any}()
    if flag_mean_DP == true
        X_minus_g = (X[1]-g[1])
        for i = 2:size(X,1)
            X_minus_g = X_minus_g + (X[i]-g[i])
        end
        X_minus_g = X_minus_g / size(X,1)
        input_arr = create_input_rgb_xy_mask(X_minus_g, mask)
        push!(pts, 1 => input_arr)
    else
        pts = Dict{Any,Any}()
        for i = 1:size(X,1)
            X_minus_g = X[i]-g[i]
            input_arr = create_input_rgb_xy_mask(X_minus_g, mask)
            push!(pts, i => input_arr)
        end
    end
    return pts
end


function results_HDP_to_μ_idx(X, model_results, color_means, mask, flag_mean_DP)
    μ_idx_list = []

    for img = 1:size(X,1)
        size_1, size_2, size_3 = size(X[img])
        if flag_mean_DP == true
            labels = model_results[1] #for mean DP
        else
            labels = model_results[img]
        end
        μ_idx = zeros(size(X[img]))
        idx = 1
        for i=1:size_2
            for j=1:size_3
                if mask[i,j] > 0
                    μ_idx[:, i, j] .= color_means[labels[idx]]
                    idx = idx + 1
                else
                    μ_idx[:, i, j] .= [1, 1, 1]
                end
            end
        end
        push!(μ_idx_list, μ_idx)
    end
    return μ_idx_list
end

function results_HDP_to_label_idx(X, model_results)
    idx_list = []

    for img = 1:size(X,1)
        size_1, size_2, size_3 = size(X[img])
        labels = model_results[img]
        label_idx = zeros(size_2,size_3)
        for i=1:size_2
            for j=1:size_3
                label_idx[i, j] = labels[(i - 1) * size_3 + j]
            end
        end
        push!(idx_list, label_idx)
    end
    return idx_list
end

# can move it two to superpixles - no need probobly !
function Sample_Reflectance(X, g, scalar_g, scalar_l, ν_g, v_l, mask, flag_mean_DP, λ_g, λ_l)
    gdim=3
    println("Sample Reflectance")

    images_dict =  create_images_mean_input(X, g, 5, mask, flag_mean_DP)

    global_hyper_params, local_hyper_params = create_priors_hdp(gdim, 2, images_dict, scalar_g, scalar_l, ν_g, v_l, λ_g, λ_l) #for vhdp!!!
    iters = 100
    #function vhdp_fit(data,gdim, α, γ, η, gprior::distribution_hyper_params, lprior, iters, initial_custers = 1, burnout = 5)
    α = 10  
    γ = 1 
    η = 10  
    hdp, history = vhdp_fit(images_dict, gdim, α, γ, η, global_hyper_params, local_hyper_params, iters)
    color_means = [(x.cluster_params.cluster_params.distribution.μ)[1:3] for x in hdp.global_clusters]
    if flag_mean_DP == true
        vhdpmm_global = Dict([i=> create_global_labels(hdp.groups_dict[i]) for i=1:1])
    else
        vhdpmm_global = Dict([i=> create_global_labels(hdp.groups_dict[i]) for i=1:size(X,1)])
    end

    #color_covs = [(x.cluster_params.cluster_params.distribution.Σ)[1:3, 1:3] for x in hdp.global_clusters]
    #vhdpmm_local = Dict([i=> hdp.groups_dict[i].labels for i=1:size(X,1)])

    μ_idx_list = results_HDP_to_μ_idx(X, vhdpmm_global, color_means, mask, flag_mean_DP)

    #labels = results_HDP_to_label_idx(X, vhdpmm_global) #μ_idx_list, color_covs, labels, size(color_means,1)
    return μ_idx_list
end


function X_minus_g_to_vec(X_minus_g)
    vec_size = size(X_minus_g,2) * size(X_minus_g,3)
    X_minus_g_vec = zeros(vec_size * 3, 1)
    X_minus_g_vec_3 = zeros(vec_size, 3)
    for i = 1:3
        X_minus_g_vec[1 + (i-1) * vec_size : i * vec_size] .= vec(X_minus_g[i,:,:])
        X_minus_g_vec_3[:, i] .= vec(X_minus_g[i,:,:])
    end
    return vec(X_minus_g_vec), X_minus_g_vec_3
end



function create_A_3_3(size_1, size_2)
    A = spzeros(size_1*size_2*8, size_1*size_2)
    counter = 1 # A rows counter

    # left and right neighbors loop
    for i = 1:(size_1*size_2)-size_1
        # right neighbors
        A[counter, i] = 1
        A[counter, i+size_1] = -1
        counter = counter + 1

        #left neighbors
        A[counter, i] = -1
        A[counter, i+size_1] = 1
        counter = counter + 1
    end

    # up and down neighbors loop
    for j = 0:size_2-1
        for i = 2:size_1
            # up neighbors
            A[counter, i+(j*size_1)] = 1
            A[counter, (i-1)+(j*size_1)] = -1
            counter = counter + 1

            # down neighbors
            A[counter, i+(j*size_1)] = -1
            A[counter, (i-1)+(j*size_1)] = 1
            counter = counter + 1
        end
    end

    # up left and down right
    for j = 0:size_2-2
        for i = size_1+2:(2*size_1)
            # up left
            A[counter, i+(j*size_1)] = 1
            A[counter, (i-1)+((j-1)*size_1)] = -1
            counter = counter + 1

            # down right
            A[counter, i+(j*size_1)] = -1
            A[counter, (i-1)+((j-1)*size_1)] = 1
            counter = counter + 1
        end
    end

    # up right and down left
    for j = 0:size_2-2
        for i = 2:size_1
            # up right
            A[counter, i+(j*size_1)] = 1
            A[counter, i+((j+1)*size_1)-1] = -1
            counter = counter + 1

            # down right
            A[counter, i+(j*size_1)] = -1
            A[counter, i+((j+1)*size_1)-1] = 1
            counter = counter + 1
        end
    end

    return A
end

function mask_X_minus_g(X_minus_g, mask)
    size_1, size_2, size_3 = size(X_minus_g)
    X_minus_g_new = zeros(size(X_minus_g))
    idx = 1
    for i=1:size_2
        for j=1:size_3
            if mask[i,j] > 0
                X_minus_g_new[:, i, j] .= X_minus_g[:, i, j]
                idx = idx + 1
            else
                X_minus_g_new[:, i, j] .= [1, 1, 1]
            end
        end
    end
    return X_minus_g_new
end

function LS_Shading(Log_X, μ_idx, Ax, mask)
    println("Sample Shading Global")
    num_of_images = size(Log_X,1)
    g_list = []

    for i = 1:num_of_images
        X_minus_g = Log_X[i] - μ_idx[i] #X[i] - exp.(μ_idx[i]) #Log_X[i] - μ_idx[i]
        X_minus_g = mask_X_minus_g(X_minus_g, mask)

        X_minus_g_vec, X_minus_g_vec_3 = X_minus_g_to_vec(X_minus_g) #this is the y



        g = Ax\X_minus_g_vec_3

        res_R = reshape(g[:,1], size(X_minus_g,2), size(X_minus_g,3))
        res_G = reshape(g[:,2], size(X_minus_g,2), size(X_minus_g,3))
        res_B = reshape(g[:,3], size(X_minus_g,2), size(X_minus_g,3))


        color_image = zeros(3,size(X_minus_g,2), size(X_minus_g,3))
        color_image[1,:,:] .= res_R
        color_image[2,:,:] .= res_G
        color_image[3,:,:] .= res_B

        println("shading" * string(i) * " OK")
        push!(g_list, color_image)
    end
    return g_list
end



function LS_Shading_gray(Log_X, μ_idx, Ax, mask)
    println("Sample Shading Global")
    num_of_images = size(Log_X,1)
    g_list = []

    for i = 1:num_of_images
        X_minus_g = Log_X[i] - μ_idx[i] #X[i] - exp.(μ_idx[i]) #Log_X[i] - μ_idx[i]
        X_minus_g = mask_X_minus_g(X_minus_g, mask)
        X_minus_g_mean = (X_minus_g[1,:,:] .+ X_minus_g[2,:,:] .+ X_minus_g[3,:,:]) ./ 3

        X_minus_g_vec = vec(X_minus_g_mean) #this is the y

        g = Ax\X_minus_g_vec

        g_reshape = reshape(g, size(X_minus_g,2), size(X_minus_g,3))

        color_image = zeros(3,size(X_minus_g,2), size(X_minus_g,3))
        color_image[1,:,:] .= g_reshape
        color_image[2,:,:] .= g_reshape
        color_image[3,:,:] .= g_reshape

        #save("/vildata/dorlitvak/img_decomp/test_images/color_image.png", colorview(RGB, map(clamp01nan, exp.(color_image))))
        println("shading" * string(i) * " OK")
        push!(g_list, color_image)
    end
    return g_list
end



############################ M A I N ############################
function save_A(A_path, Log_X)
    size_1 = size(Log_X[1],2)
    size_2 = size(Log_X[1],3)
    A = create_A_3_3(size_1, size_2)
    f = open(A_path,"w")
    serialize(f, A)
    close(f)
    return A
end



# X and SP are lists of images and their correspond SP.
function II_via_MCMC(X, Log_X, scalar_g, scalar_l, ν_g, v_l, λ_g, λ_l, λ, A_path, mask, flag_mean_DP, flag_gray_LS, flag_create_mat, flag_use_mask)
    #flags
    #flag_mean_DP = false #false - no mean, ture - with mean
    #flag_gray_LS = true #false - RGB color, true - grayscale color
    #flag_create_mat = false #false - only read matrix, true - create matrix
    #flag_use_mask = true #true - use mask, false - do not use mask

    if flag_use_mask == false
        mask = ones(size(X[1]))
    end

    g, μ_idx = Initialize_lists(Log_X)
    #λ = 1 #0.3
    size_1 = size(Log_X[1],2)
    size_2 = size(Log_X[1],3)
    A = spzeros(size_1*size_2*8, size_1*size_2)

    if flag_create_mat == true
        A = save_A(A_path, Log_X)
    end

    f1 = open(A_path)
    A = deserialize(f1)
    close(f1)

    Id = SparseArrays.sparse(I, size_1*size_2, size_1*size_2)
    Ax = (Id' * Id) + (λ * A' * A)


    num_of_clusters = 0
    it = 1
    start = time()
    while(it < 4)
        μ_idx = Sample_Reflectance(Log_X, g, scalar_g, scalar_l, ν_g, v_l, mask, flag_mean_DP, λ_g, λ_l) #, Σ_x_clusters, global_labels, num_of_clusters
        
        if flag_gray_LS == false
            g = LS_Shading(Log_X, μ_idx, Ax, mask) #use RGB mode
        else
            g = LS_Shading_gray(Log_X, μ_idx, Ax, mask) #use Gray mode
        end
        
        println("finish II_via_MCMC it: ", it)
        it = it + 1

    end
    elapsed = time() - start
    println("-----------------FINIAH - STATISTICS:--------------------")
    println("Run Time in min:", elapsed/60)
    println("Number of total clusters:", num_of_clusters)
    return μ_idx, g
end
