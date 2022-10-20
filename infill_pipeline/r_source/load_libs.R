is.installed = function(mypkg){
    is.element(mypkg, installed.packages()[,1])
} 

# May need to unlock
# Sys.setenv(R_INSTALL_STAGED = FALSE)

r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)

#######################
n = "parallel"
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'ggplot2'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'dplyr'
if (!is.installed(n)) {
    install.packages(n)
}
suppressPackageStartupMessages(library(n, character.only = TRUE))

n = 'rsm'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'desirability'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'gridExtra'
if (!is.installed(n)) {
    install.packages(n)
}
suppressPackageStartupMessages(library(n, character.only = TRUE))

n = 'GPareto'
if (!is.installed(n)) {
    install.packages(n)
}
suppressPackageStartupMessages(library(n, character.only = TRUE))

n = 'DiceDesign'
if (!is.installed(n)) {
    install.packages(n)
}
suppressPackageStartupMessages(library(n, character.only = TRUE))

n = 'R.matlab'
if (!is.installed(n)) {
    install.packages(n)
}
suppressPackageStartupMessages(library(n, character.only = TRUE))

n = 'NbClust'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'R6'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'purrr'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'pracma'
if (!is.installed(n)) {
    install.packages(n)
}
suppressPackageStartupMessages(library(n, character.only = TRUE))

n = 'readr'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'pso'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'pryr'
if (!is.installed(n)) {
    install.packages(n)
}
suppressPackageStartupMessages(library(n, character.only = TRUE))

n = 'yaml'
if (!is.installed(n)) {
    install.packages(n)
}
suppressPackageStartupMessages(library(n, character.only = TRUE))

n = 'nat.utils'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)

n = 'openssl'
if (!is.installed(n)) {
    install.packages(n)
}
library(n, character.only = TRUE)


