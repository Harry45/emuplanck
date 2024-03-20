library(lhs)
# nlhs <- 500
# dimensions <- 3

args <- commandArgs(trailingOnly = TRUE)

cat('Number of LHS points is:', args[1], "\n")
cat('Number of dimensions is:', args[2], "\n")

n = args[1]
dimensions = args[2]

lhs_points <- maximinLHS(n, dimensions)

# filename
file <- paste("lhs/", "samples_", as.character(dimensions), "_",
    as.character(n), ".csv",
    sep = ""
)

# write output
write.csv(lhs_points, file)
