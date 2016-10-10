options(stringsAsFactors=F)

x <- read.csv('./models/sup-no-thread.vec', header=FALSE, sep=' ', skip = 1, quote='')
x <- x[order(x[,1]),-ncol(x)]

plot(x[,c(2, 3)], cex=0.2, col=1 + grepl('_e$', x[,1]))


x <- read.csv('./models/dev.vec', header=FALSE, sep=' ', skip = 1, quote='')
x <- x[order(x[,1]),-ncol(x)]

plot(x[,c(2, 3)], cex=0.2, col=1 + grepl('_s$', x[,1]))



# --
options(stringsAsFactors=F)

while(TRUE) {
    # x <- read.csv('./big-bilum.hist', header=FALSE, sep='|', skip=40000)
    x <- read.csv('./vbig.hist', header=FALSE, sep='|')
    names(x) <- c('model', 'progress', 'lr', 'loss')

    plot(x$loss, type='h', col=as.factor(x$model))
    Sys.sleep(0.1)
}

