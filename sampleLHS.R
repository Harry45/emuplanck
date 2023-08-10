# setwd('/home/harry/Documents/Oxford/Astrophysics/Projects/MOPED-GP-Expansion/comgp/')
library(lhs)
nlhs = seq(1500, 5000, by = 500)
d = 6

for (n in nlhs){
	X = maximinLHS(n, d)

	# filename
	file = paste('lhs/', 'samples_', as.character(d), '_', as.character(n), '.csv', sep ='')

	# write output
	write.csv(X, file)
}
