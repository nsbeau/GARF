# Nicole Beauregard
# GARF Model

#=
Copyright 2020 University og Connecticut

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=#

# Import Libraries
using DecisionTree
using DataFrames, CSV, DelimitedFiles, GLM, Random
using Plots

# Import Feature and label Training Data
featuredata = CSV.read("featuredata.csv")
labeldata = CSV.read("labeldata.csv")

# Import randomized MOF list
mofdf = CSV.read("mofnum.csv")
mofmatrix = convert(Matrix, mofdf)
mofnum = mofmatrix[:,1]

# Calculate Training Number
trainperc = .1                                                      # Training percent
trainnum = Int(ceil(length(featuredata[:,1])*trainperc))            # Number to Train
# Column Number of Label Data to Train On
prop = 2                                                            # Mass Based Uptake

# Convert Feature and Labels from Data Frames to Matrices
features = convert(Matrix, featuredata[mofnum[1:trainnum],2:13])    # First column is MOF number, so we exclude it
labels = convert(Vector, labeldata[mofnum[1:trainnum],prop])

# All MOFs and all labels
allmofs = convert(Matrix, featuredata[mofnum,2:13])
alllabels = convert(Vector, labeldata[mofnum,prop])


# Build Model to train on 12 features for Mass Baked Methane Uptake
model = build_forest(labels, features, 7, 250)                      # Train on 7/12 features, 250 trees

# Make an array filled with the realistic ranges for each property value
# Each range is the min and max of the values in the hMOF database [min, max]
function propertyrangesMOF()
    # Make empty array which is the size of the number of features
    R = Array{Any}(undef,12)
    # Structural Properties
    R[1] = [0.05, .97]      # void fraction
    R[2] = [0.0, 24.75]     # PLD
    R[3] = [0.0, 24.75]     # LCD
    R[4] = [0.0, 6947.0]    # surface area
    R[5] = [0.12, 4.04]     # density
    # Chemical Properties
    R[6] = [6.0, 565.0]     # degree unsaturated
    R[7] = [0.19, 1.36]     # degree unsaturated per carbon
    R[8] = [0.85, 50.0]     # metal percentage
    R[9] = [0.35, 0.62]     # electronegativity
    R[10] = [0.12, 2.15]    # weighted electronegativity
    R[11] = [0.0, 6.5]      # nitrogen over oxygen
    R[12] = [6.5, 51.0]     # oxygen to metal ratio

    # Type: 0 = Float, 1 = Integer, 2 = Metal (In this case, all are Float)
    T = [0 0 0 0 0 0 0 0 0 0 0 0]
    # Return Ranges and Types
    return R, T
end

# Make initial population of chromosomes
function initialpopulation(popsize,R,T)
    # Make empty array which is the size of the population
    initpop = Array{Any}(undef,popsize)
    # For each member of the population
    for i = 1:popsize
        # The chromosome is the size of the number of properies
        chrom = zeros(length(R))
        # For each feature
        for j = 1:length(R)
            # Extract the property range for that feature
            range = R[j]
            # For property 3, largest cavity diameter
            if j == 3
                # The smallest the LCD can be is the size of the PLD which is property 2
                range[1] = chrom[2]     # Replace the minumum LCD value with the PLD value
            end
            # Randomly select initial value from the range using rng
            chrom[j] = rand()*(range[2]-range[1])+range[1]
        end
        # Place entire chromosome back in the initial population array
        initpop[i] = chrom
    end
    # Return initial population
    return initpop
end

# Apply random forest
function rf(model,properties)
    # Predition needs to apply the model and the chomosome full of properties
    prediction = apply_forest(model, properties)
    # Return prediction
    return prediction
end

# Calculate fitnesses
function fitness(model,chrompop)
    # Make an array of zeros of the population size
    fitnesses = zeros(length(chrompop))
    # For each member of the population
    for i = 1:length(chrompop)
        # Extract the chromosome for that member of the population
        props = chrompop[i]
        # Use those properties to predict methane uptake
        fit = apply_forest(model, props)
        # Place that fitness in the array of fitnesses for the entire population
        fitnesses[i] = fit[1]
    end
    # Return fitness array
    return fitnesses
end

# Find elites
function eliteselection(fitnesses,chrompop,elitenum)
    # Sort fitnesses from largest to smallest value to find location of elites in array
    index = sortperm(fitnesses, rev=true)
    # Use the index to to extract the elite fitness values from the fitness array
    elitefit = fitnesses[index[1:elitenum]]
    # Use the index to to extract the elite chromosomes from the population
    elitechrom = chrompop[index[1:elitenum]]
    # Return the elite fitnesses and chromosomes
    return elitefit, elitechrom
end

# Tournament seletcion method to choose parents
function tournament(popsize,chrompop,fitnesses)
    # Find the number of contestants
    numcont = convert(Int64,ceil(popsize*0.2))      # Number = 20% of population
     # Randomly select the contestants (Gives location index)
    contestants = randperm(popsize)[1:numcont]
    # Extract the fitnesses and chromosomes of contestants
    cfit = fitnesses[contestants]
    cchrom = chrompop[contestants]
    # Sort the fitnesses of the contestants (Gives location index)
    sortedfit = sortperm(cfit,rev=true)
    # Parents are the two chromosomes with the highest fitnesses (Use index to extract)
    p1 = cchrom[sortedfit[1]]
    p2 = cchrom[sortedfit[2]]
    # Return parents
    return p1, p2
end

# Recombine parents
function recomb(p1,p2,recombpercent)
    # Randomly choose number between 0 and 1
    p = rand()
    # If p > recombination percent, children become parents, no recombination
    if p > recombpercent
        child1 = copy(p1)
        child2 = copy(p2)
    # If less, begin recombination
    else
        # Chromosome length is the length of the parent
        clength = length(p1)
        # Copy children as parents, prevents accidental overwrite
        child1 = copy(p1)
        child2 = copy(p2)
        # Randomly choose swap location
        swap = rand(1:clength)
        # Extract chromosomes from 1:swap point
        p1swap = p1[1:swap]
        p2swap = p2[1:swap]
        # Replace chromosome 1:swap point in each chromsome with the other chromosome
        child1[1:swap] = p2swap
        child2[1:swap] = p1swap
    end
    # Return children
    return child1, child2
end

# Mutation children
function mutation(child, mutper, R, T)
    # Randomly choose number between 0 and 1
    p = rand()
    # If p is greater than the mutation rate, do not mutate
    if p > mutper
        childnew = copy(child)
    # If less, begin mutation
    else
        # Chromosome length is the length of the child
        clength = length(child)
        # Randomly choose mutation point
        mutswap = rand(1:clength)
        # Extract property range of mutation point
        range = R[mutswap]
        # Randomly select new value from the range
        child[mutswap] = rand()*(range[2]-range[1])+range[1]
    end
    # Return child
    return child
end

# Create new generation
function getnewgen(chrompop, fitnesses, recombperc, mutperc, elitefrac, R, T)
    # Popuation size is the size of the chromosome array
    popsize = length(chrompop)
    # Make an empty array of population size
    newgen = Array{Any}(undef,popsize)
    # Calculate elite number using the elite fraction, ensure it is an integer
    elitenum = convert(Int64,length(fitnesses)*elitefrac)
    # Find elite fitnesses and elite fractions
    elitefits, elites = eliteselection(fitnesses,chrompop,elitenum)
    # Place elite chromosomes in the new generation array
    newgen[1:elitenum] = elites
    # Start count 1 after the elite number
    count = elitenum + 1
    # For the rest of the population
    for i = 1:ceil((popsize-elitenum)/2)
        # Select parents
        p1, p2 = tournament(popsize,chrompop,fitnesses)
        # Use parents to recombine and make children
        child1, child2 = recomb(copy(p1),copy(p2),recombperc)
        # Mutate the children
        newgen1 = mutation(copy(child1), mutperc, R, T)
        newgen2 = mutation(copy(child2), mutperc, R, T)
        # Add first child to new generation array
        newgen[count] = copy(newgen1)
        # If there is still room in the array, add the second child
        if count < popsize
            newgen[count+1] = copy(newgen2)
        end
        # Increase count by 2
        count += 2
    end
    # Return new generation
    return newgen
end


function ga_MOF(popsize,generations, elitefrac,recombperc,muteperc,model)
    # Extract property ranges
    R, T = propertyrangesMOF()
    # Calculate initial initialpopulation
    chrompop = initialpopulation(popsize,R,T)
    # Calculate fitnesses of the initial population
    fitnesses = fitness(model,chrompop)
    # Make an array to hold the best fitnesses (+1 gen to account for intial population)
    bestfit = zeros(generations+1)
    # Find the best fitness and add it to the array
    bestfit[1] = maximum(fitnesses)
    # For each generation
    for i = 1:generations
        # Calculate new generation
        newgen = getnewgen(copy(chrompop), copy(fitnesses), recombperc, muteperc, elitefrac, R, T)
        # Calculate the fitness of the new generation
        newfit = fitness(model,copy(newgen))
        # Find the best fitness and add it to the array
        bestfit[i+1] = maximum(newfit)
        # Copy new gen as new chrompop
        chrompop = copy(newgen)
        # Copy new fit as fitness
        fitnesses = copy(newfit)
    end
    # Sort the final fitness
    sortedfit = sortperm(fitnesses,rev=true)
    # Use index to order final fitnesses
    finalfit= fitnesses[sortedfit]
    # Use index to order final chromosomes
    finalchrom = chrompop[sortedfit]
    # Return final chromosome, fitness, and best fitness
    return finalchrom, finalfit, bestfit
end

# Find the real mof matches
function findrealmofs(allmofs,chrompop,elites,tolerance=.05)
    # Make array of size elites
    closestMOFs = Array{Any}(undef,elites)
    # Array for number of matches
    nummatches = zeros(elites)
    # For each elites
    for i = 1:elites
        # Find tolerance
        tol = tolerance
        # Extract elite chromosome
        elitechrom = chrompop[i]
        # Create test matrix for all properties of each hMOF
        testmatrix = zeros(130398,12)
        # Create matrix for the sum of points
        sums = zeros(130398)
        # For each property
        for j = 1:12
            # Find where property is within 5% of hMOF property
            loc = findall(x-> (elitechrom[j]*(1-tol)) < x <(elitechrom[j]*(1+tol)), allmofs[:,j])
            # If there are properties that match
            if length(loc) > 0
                # For each matching location
                for l = 1:length(loc)
                    # Add a one (point) to the test matrix
                    testmatrix[loc[l],j] = 1
                end
            end
        end
        # For each hMOF
        for j = 1:130398
            # Sum point totals for each hMOF
            sums[j] = sum(testmatrix[j,:])
        end
        # If there are point totals greater than zero
        if maximum(sums) > 0
            # Find the macthes that have the highest point total
            bestmatches = findall(x-> x==maximum(sums), sums)
            # Cloest mofs for this elite add to array
            closestMOFs[i] = bestmatches
            # Add the number of matches to array
            nummatches[i] = maximum(sums)
        end
    end
    # Return cloest mofs and number of matches
    return closestMOFs,nummatches
end

# Least Squares
function leastsquares(allmofs,chrompop,elites)
    # Create array to hold all values of properties for each emof
    allvals = zeros(length(allmofs[:,1]),elites)
    # Array to hold sorted locations
    sortedlocs = zeros(length(allmofs[:,1]),elites)
    # Normalize array
    newmofs = Array{Any}(undef,130398,12)
    # Place holder array for maximum value
    maxvec = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # For each property
    for i = 1:12
        # Find max of each property
        max = maximum(allmofs[:,i])
        # Divide each property by the max value
        newmofs[:,i] = allmofs[:,i]/max
        # Add max value to array
        maxvec[i] = max
    end
    # For each elite
    for i = 1:elites
        # Divide chromosome by the max value
        chrom = chrompop[i]./maxvec
        # Create array to hold scores
        score = zeros(length(allmofs[:,1]))
        # For each hMOF
        for j = 1:length(allmofs[:,1])
            # Calculate euclidean distance
            dist = sqrt.((allmofs[j,:] - chrom).^2)
            # Score is the sum
            score[j] = sum(dist)
        end
        # Put score array for eMOF in total array
        allvals[:,i] = score
        # Sort the scores
        sortloc = sortperm(score,rev=true)
        # Put sorted locations in array
        sortedlocs[:,i] = sortloc
    end
    return allvals, sortedlocs
end

# Find best matches
function findmatches(closestMOFs,bestrealmofs,elites,num=1000)
    # Create array for matches
    matches = Array{Any}(undef,elites)
    # For each elite
    for i = 1:elites
        # Extracted the predicted hMOFs
        predmofs = closestMOFs[i]
        # Empty array to hold matches
        match = []
        # For each predicted mof
        for p = 1:length(predmofs)
            # Check if that predicted MOF is in the top 1000 best MOFs
            loc = findall(x-> x==predmofs[p], bestrealmofs[1:num])
            # If there are matches
            if length(loc) > 0
                # Add MOFs to matches array
                push!(match,predmofs[p])
            end
        end
        # If there are matches
        if length(match) > 0
            # Add matches to array
            matches[i] = match
        else
            # If not, matches is zero
            matches[i] = 0
        end
    end
    # Return matches
    return matches
end


# Least squares 2
function leastsquares(allmofs,chrompop,elites)
    # Create array to hold all values of properties for each emof
    allvals = zeros(length(allmofs[:,1]),elites)
    # Array to hold sorted locations
    sortedlocs = zeros(length(allmofs[:,1]),elites)
    # For each elite
    for i = 1:elites
        # Extract chromosome for that elite
        chrom = chrompop[i]
        # Make array to hold scores for each hMOF
        score = zeros(length(allmofs[:,1]))
        # For each hMOF
        for j = 1:length(allmofs[:,1])
            # Initial sum = 0
            sum = 0
            # For each property value in the chromosome
            for k = 1:length(chrom)
                # Calculate that properties euclidean distance (divide by max value to normalize)
                test = ((chrom[k]-allmofs[j,k])/maximum(allmofs[:,k]))^2
                # Add value to sum
                sum += test
                # Set test = 0
                test = 0
            end
            # Score is the square root of the sum
            score[j] = sqrt(sum)
        end
            # Add score to total array
        	allvals[:,i] = score
            # Sort scores
            sortloc = sortperm(score,rev=true)
            # Add sorted locations to total array
            sortedlocs[:,i] = sortloc
    end
    # Return all scores and all sorted locations
    return allvals,sortedlocs
end
