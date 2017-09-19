# spmvharness

## To build:

(in project root)

	mkdir build
	cd build
	cmake ..
	make

## Run an example

(in project root)

	cd scripts/example
	./runexample.sh

# The algorithms

## Sparse matrix dense vector multiplication

Floating point sparse matrix, dense vector multiplication. Single iteration per trial.

## Single Source Shortest Paths

Iterative linear algebraic implementation of Single Source Shortest Paths. We initialise our linear algebra like so:

Initial vector X = infinity everywhere (apart from root. which is zero)
Initial vector Y = (same) 

Alpha = 0.0f  
Beta = 0.0f

zero = float max

addition = floating point minimum
multiplication = floating point addition

## Breadth first search

Iterative linear algebraic implementation of breadth first search. Linear algebra initialised like so:

(we use Integer types for levels)

Initial vector X = zero everywhere (aside from root, which is one)
Initial vector X = zero everywhere (aside from root, which is one)

Alpha = 1
Beta = 0

zero = 0
one = 1

addition = boolean "or"
multiplication = boolean "and"

## Single source shortest paths

(todo)

## Pagerank

(todo)