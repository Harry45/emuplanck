library(lhs)
nlhs <- 2000
dimensions <- 6

for (n in nlhs) {
    lhs_points <- maximinLHS(n, dimensions)

    # filename
    file <- paste("lhs/", "samples_", as.character(dimensions), "_",
        as.character(n), ".csv",
        sep = ""
    )

    # write output
    write.csv(lhs_points, file)
}
