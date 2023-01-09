using Distributed
using Statistics
using LinearAlgebra
using PDMats
using SpecialFunctions
using FFTW
using DSP


addprocs(10)

#For the HDP, vHDP version
@everywhere using VersatileHDPMixtureModels

# For the DP version
@everywhere using DPMMSubClusters


# gaussian process kernel function
# from: http://krasserm.github.io/2018/03/19/gaussian-processes/
function exp_kernel(r, l=1.0, sigma_f=1.0)
    return (sigma_f ^ 2) * exp(-( r / (2*(l^2)) ) )
end


function matern_kernel(r, l=1.0, sigma_f=1.0)
    r = abs(r)
    if r == 0
        r = 1e-8
    end
    part1 = (2 ^ (1 - sigma_f)) / gamma(sigma_f)
    part2 = (sqrt(2 * sigma_f) * r / l) ^ sigma_f
    part3 = besselk(sigma_f, (sqrt(2 * sigma_f) * r) / l)
    return part1 * part2 * part3
end


function kernel_mat(X1, X2, l=1.0, sigma_f=1.0)
    cov = zeros((size(X1,1), size(X2,1)))
    for i in 1:size(X1,1)
        for j in 1:size(X2,1)
            xi = X1[i,:]
            xj = X2[j,:]
            xixj = (xi-xj)' * (xi-xj)
            cov[i,j] = exp_kernel(xixj, l, sigma_f)
        end
    end
    return cov
end


function kernel_cov_grid(size_1, size_2, size_3, l=1.0, sigma_f=1.0)
    #X_grid = [[z, x , y] for x in 1:size_2, y in 1:size_3, z in 1:size_1]
    X_grid = [[z, y, x] for x in 1:size_2, y in 1:size_3, z in 1:size_1]
    X_grid_reshape = Int64.(hcat(X_grid...)')

    cov = kernel_mat(X_grid_reshape, X_grid_reshape, l, sigma_f)
    cov = Symmetric(cov)
    return cov
end


function Initialize(X)
    g = zeros(size(X))
    #z = zeros(size(X))
    μ_idx = zeros(size(X))
    println("Done Initialize")
    return g, μ_idx
end


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
        g_reshape[i+1,:,:] = reshape(one_color, x_size, y_size)'
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

#This hyper parameters DOES take the x,y asix into account and the RGB values!!!!
function create_hupter_params_xy(input_arr)

    rgb_prior_multiplier = 30.0 #TEST:
    xy_prior_multiplier = 1.0 #TEST:

    data_cov = cov(input_arr')
    data_cov[4:5,1:3] .= 0
    data_cov[1:3,4:5] .= 0

    data_cov[1:3,1:3] .*= rgb_prior_multiplier
    data_cov[4:5,4:5] .*= xy_prior_multiplier

    data_mean = mean(input_arr,dims = 2)[:]

    hyper_params = DPMMSubClusters.niw_hyperparams(1.0, data_mean, 8, data_cov)
    return hyper_params
end

#This hyper parameters doesn't take the x,y asix into account!
function create_hupter_params_GRB(input_arr)

    #rgb_prior_multiplier = 30.0 #TEST:
    #xy_prior_multiplier = 1.0 #TEST:

    data_cov = cov(input_arr')
    #data_cov[4:5,1:3] .= 0
    #data_cov[1:3,4:5] .= 0

    #data_cov[1:3,1:3] .*= rgb_prior_multiplier
    #data_cov[4:5,4:5] .*= xy_prior_multiplier

    data_mean = mean(input_arr,dims = 2)[:]

    hyper_params = DPMMSubClusters.niw_hyperparams(1.0, data_mean, 8, data_cov)
    return hyper_params
end

# ldim is 0 in the hdp part
function create_priors_hdp(gdim,ldim,images_dict)
    data_cov = cov(images_dict[1]')
    data_mean_1 = mean(images_dict[1],dims = 2)[:]
    data_mean_2 = mean(images_dict[2],dims = 2)[:]
    data_mean_3 = mean(images_dict[3],dims = 2)[:]

    mean_data = (data_mean_1 + data_mean_2 + data_mean_3) / 3
    #g_prior = niw_hyperparams(1.0, zeros(gdim), gdim+3, Matrix{Float64}(I, gdim, gdim)*1)
    g_prior = niw_hyperparams(1.0, mean_data, 8, Matrix{Float64}(I, gdim, gdim)*2)
    l_prior = niw_hyperparams(1.0, zeros(ldim), ldim+3, Matrix{Float64}(I, ldim, ldim)*1)

    return g_prior, l_prior
end


function create_symetric_Σ_x(gdim, x1, x2)
    Σ_x = zeros((gdim,gdim))
    #x1 = 0.1/(25500) # TEST: 0.1/(255)
    #x2 = 0.01/(25500) # TEST: 0.01/(255)
    for i in 1:gdim
        for j in 1:gdim
            if i == j
                Σ_x[i,j] = x1
            else
                Σ_x[i,j] = x2
            end
        end
    end
    return Σ_x
end

function create_images_input(X, g, gdim)

    pts = Dict{Any,Any}()
    for i = 1:size(X,1)
        X_minus_g = X[i]-g[i]
        input_arr = create_input_rgb(X_minus_g)
        #input_arr = vec(X[i]-g[i])'
        push!(pts, i => input_arr)
    end
    return pts
end

function results_HDP_to_μ_idx(X, model_results, color_means)
    μ_idx_list = []

    for img = 1:size(X,1)
        size_1, size_2, size_3 = size(X[img])
        labels = model_results[img]
        μ_idx = zeros(size(X[img]))
        for i=1:size_2
            for j=1:size_3
                μ_idx[:, i, j] = color_means[labels[(i - 1) * size_3 + j]]
            end
        end
        push!(μ_idx_list, μ_idx)
    end
    return μ_idx_list
end



# can move it two to superpixles - no need probobly !
function Sample_Reflectance(X, g, gdim=3)
    println("Sample Reflectance")
    images_dict = create_images_input(X, g, gdim)
    #gprior, lprior = create_default_priors(gdim,0,:niw)
    gprior, lprior = create_priors_hdp(gdim,0,images_dict)
    iters = 150
    model = hdp_fit(images_dict, 10, 1, gprior, iters)
    model_results = get_model_global_pred(model[1])
    color_means = [(x.cluster_params.cluster_params.distribution.μ) for x in model[1].global_clusters]
    μ_idx_list = results_HDP_to_μ_idx(X, model_results, color_means)

    return μ_idx_list
end


function kernel_cov_grid_superpixel(sp_idx_int, l=1.0, sigma_f=1.0)
    sp_idx_int_RGB = zeros(size(sp_idx_int, 1) * 3, 3)
    for i = 1:3
        sp_idx_int_RGB[1 + (i-1) * size(sp_idx_int, 1) : i * size(sp_idx_int, 1), 1] .= i
        sp_idx_int_RGB[1 + (i-1) * size(sp_idx_int, 1) : i * size(sp_idx_int, 1), 2:3] .= sp_idx_int
    end
    cov = kernel_mat(sp_idx_int_RGB, sp_idx_int_RGB, l, sigma_f)
    cov = Symmetric(cov)
    return cov
end


function Sample_Shading_per_SP(X, μ_idx, Σ_x, SP, l=1.0, sigma_f=1.0)
    println("Sample Shading per SP")

    X_minus_r = X-μ_idx
    full_g = zeros(size(X_minus_r))
    SP_list = unique(SP)
    it = 1
    Threads.@threads for superpixel in SP_list
        println("superpixel ", superpixel, " it ", it, " out of ", size(SP_list,1))

        it = it + 1
        # sp_idx: The superpixel pixels indexes in CartesianIndex [number_of_SP*2]
        sp_idx = findall(z->z==superpixel, SP)
        # sp_idx_int: The superpixel pixels indexes in Int64 array [number_of_SP*2]
        sp_idx_int = hcat(getindex.(sp_idx, 1), getindex.(sp_idx,2))
        Σ_g = kernel_cov_grid_superpixel(sp_idx_int, l, sigma_f) #TEST: l, sigma_f

        # X_superpixel: The pixels color in the superpixel RGB [number_of_SP*3]
        X_superpixel = X_minus_r[:, sp_idx]
        id_n = Matrix(1I, size(X_superpixel,2), size(X_superpixel,2))
        k = kron(Σ_x, id_n)
        Λ_g_x = Symmetric(inv(Σ_g + k))
        mu = Σ_g' * Λ_g_x * vec(X_superpixel')
        mu_vec = vec(mu)
        sigma = Σ_g - Symmetric(Σ_g' * Λ_g_x * Σ_g)
        sigma_sym = Symmetric(sigma)
        g = rand(MvNormal(mu_vec, sigma_sym), 1)

        g = reshape(g', (size(X_superpixel,2),3))'

        full_g[:, sp_idx] .= g

    end

    return full_g
end


function Sample_Shading(X, μ_idx, Σ_x, SP, l=1.0, sigma_f=1.0)
    println("Sample Shading Global")
    num_of_images = size(X,1)
    g_list = []
    for i = 1:num_of_images
        sp_colors, sp_location = create_superpixels_array((X[i]-μ_idx[i]), SP[i])
        sp_idx_int = hcat(sp_location[1,:], sp_location[2,:])
        Σ_g = kernel_cov_grid_superpixel(sp_idx_int, l, sigma_f) #TEST: l, sigma_f

        id_n = Matrix(1I, size(sp_colors,2), size(sp_colors,2)) #$OK
        k = kron(Σ_x, id_n)
        Λ_g_x = Symmetric(inv(Σ_g + k))
        mu = Σ_g' * Λ_g_x * vec(sp_colors')
        mu_vec = vec(mu)
        sigma = Σ_g - Symmetric(Σ_g' * Λ_g_x * Σ_g)
        sigma_sym = Symmetric(sigma)
        g = rand(MvNormal(mu_vec, sigma_sym), 1)
        g = reshape(g', (size(sp_colors,2),3))'
        full_g = sp_shading_to_full_size_shading(X[i], g, SP[i])
        push!(g_list, full_g)
    end
    return g_list
end


function sp_shading_to_full_size_shading(X, g, SP)
    full_g = zeros(size(X))
    number_of_SP = size(unique(SP), 1)
    for superpixel = 1:number_of_SP
        idx = findall(z->z==superpixel, SP)
        full_g[:, idx] .= g[:, superpixel]
    end
    return full_g
end


function create_superpixels_array(X, SP)
    number_of_SP = size(unique(SP), 1)
    SP_colors = zeros((3, number_of_SP))
    SP_location = zeros((2, number_of_SP))
    for superpixel = 1:number_of_SP
        idx = findall(z->z==superpixel, SP)
        X_idx = X[:, idx]
        SP_colors[:, superpixel] = vec(mean(X_idx, dims = 2))
        sp_idx = findall(z->z==superpixel, SP) # The superpixel pixels indexes in CartesianIndex [number_of_SP*2]
        sp_idx_int = hcat(getindex.(sp_idx, 1), getindex.(sp_idx,2))
        SP_location[:, superpixel] = vec(mean(sp_idx_int, dims = 1))
    end
    return SP_colors, SP_location
end


############################ M A I N ############################

# X and SP are lists of images and their correspond SP.
function II_via_MCMC(X, SP)
    RGB = 3
    # GLOBAL GP PARAMETERS
    l_global = 20
    sigma_f_global = 0.03 #0.3 #0.7 #1
    sigma1_global = 0.01/255 #0.05/255
    sigma2_global = 0.001/255 #0.005/255

    # LOCAL GP PARAMETERS
    #l_local = 1
    #sigma_f_local = 0.3 #1
    #sigma1_local = 0.01/255
    #sigma2_local = 0.001/255


    Σ_x_global = create_symetric_Σ_x(RGB, sigma1_global, sigma2_global)
    #Σ_x_local = create_symetric_Σ_x(RGB, sigma1_local, sigma2_local)

    g, μ_idx = Initialize_lists(X)

    it = 1
    start = time()
    while(it < 2)

        μ_idx = Sample_Reflectance(X, g)

        g = Sample_Shading(X, μ_idx, Σ_x_global, SP, l_global, sigma_f_global) #g_full
        #g = g_full + Sample_Shading_per_SP(X, g_full+μ_idx, Σ_x_local, SP, l_local, sigma_f_local)

        println("finish II_via_MCMC it: ", it)
        it = it + 1
    end
    elapsed = time() - start
    println("-------------------------------------")
    println("time of the algorithm in min:", elapsed/60)
    return μ_idx, g
end
