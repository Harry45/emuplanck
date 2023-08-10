library(lhs)
nlhs <- 200
dimensions <- 6

for (n in nlhs) {
    lhs_points <- maximinLHS(n, dimensions)

    # filename
    file <- paste("lhs/", "samples_", as.character(d), "_",
        as.character(n), ".csv",
        sep = ""
    )

    # write output
    write.csv(lhs_points, file)
}
