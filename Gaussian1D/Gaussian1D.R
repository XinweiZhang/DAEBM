####################################################################################
# This file implement the Gaussian 1D Example in the paper 
# " 
####################################################################################

rm(list=ls())

set.seed(42)

##
n <- 1000
b <- 2

alp <- 3/4 # Proportion of left mode
sig <- .1  # Standard deviation of Data

ufunc0 <- function(x) {
  # Energy function of the true data generating distribution
  - log(  alp *exp(-(x+b)^2/2/sig^2) + (1-alp) *exp(-(x-b)^2/2/sig^2 ) ) 
  
}

x <- c( rnorm(alp*n, mean=-b, sd=sig), rnorm((1-alp)*n, mean=b, sd=sig) ) # generate training sample

par(mfrow=c(2,2))

hist(x, 100, xlim=c(-4,4))

ax <- seq(-5, 5, .025)

plot( ax, ufunc0(ax), type="l", ylim=c(-4,5) )

 
xi <- seq(-4, 4, .1)  # knots
p <- length(xi)

ufunc <- function(x, xi, theta) {                
  # U_theta(x)
  val <- x^2/10^2  
  grad <- 2*x /10^2  
  
  for (j in 1:length(xi)) {
    val <- val + theta[j] *ifelse(x>xi[j], x-xi[j], 0)
    grad <- grad + theta[j] *ifelse(x>xi[j], 1, 0)
  }
  
  list(val=val, grad=grad)
}

gradfunc <- function(x, xi) {       
  # Gradient of U_theta(x) with respect to x
  val <- matrix(NA, length(x), p)
  for (j in 1:length(xi))
    val[,j] <- ifelse(x>xi[j], x-xi[j], 0)
  val
}

langev <- function(z, n, eps, theta) {
  # MALA sampling
  ar.sum <- rep(0, length(z))
  
  for (i in 1:n) {
    #z <- z - eps^2/2 * ufunc(z, xi, theta)$grad + rnorm(length(z), mean=0, sd=eps)
    
    u.old <- ufunc(z, xi, theta)
    z.noise <- rnorm(length(z), mean=0, sd=1)
    
    z2 <- z - eps^2/2 * u.old$grad + eps*z.noise    
    u.new <- ufunc(z2, xi, theta)
    
    # equivalent
    #z2.noise <- (z - z2 + eps^2/2 *u.new$grad) /eps
    z2.noise <- -z.noise + (u.old$grad + u.new$grad) *eps/2
    
    u.diff <- (u.new$val -u.old$val) + z2.noise^2/2 - z.noise^2/2
    
    ar <- rep(1, length(z))
    for (j in 1:length(z)) {
      if (u.diff[j]>0)  ar[j] <- runif(1) < exp(-u.diff[j])
    }
    
    ar.sum <- ar.sum + ar
    z <- ifelse(ar, z2, z)
  }
  
  list(z=z, ar=ar.sum/n)
}

scheduler <- function(lr, iter, milestones=NULL, adj=.1){
  # Learning rate scheduling
  if(is.null(milestones)){
    lr
  }else if(iter %in% milestones){
    lr <- lr *adj
  }
  lr
}

par(mfrow=c(2,3))

niter <- 1200
lr <- 2e-1   # 

langev.n <- 10
langev.eps <- .1  

milestones <- c(400, 600, 800, 1000)

theta0 <- rnorm(p, mean=0, sd=1)  # 
theta <- theta0

z <- rnorm(1000, mean=0, sd=1) # Replay Buffer
hist(z, 100, xlim=c(-4, 4))

plot( ax, ufunc(ax, xi, theta)$val, type="l", xlim=c(-6, 6))
#abline(v=xi, lty=3)
abline(v=c(-4,-2,2,4), lty=3)

plot(0, type="n")

nsub <- 100

ar.ave <- rep(NA, niter)
uat2.diff <- rep(NA, niter)
st.dif <- rep(NA, niter)

gr <- 0

for (i in 1:niter) {
  
  theta <- theta - lr * gr
  
  sub <- sample(1:n, nsub)
  
  
  out.langev <- langev(z[sub], langev.n, langev.eps, theta) 
  z[sub] <- out.langev$z
  
  ar.ave[i] <- mean(out.langev$ar)
  
  st.dif[i] <- mean(ufunc(x[sub], xi, theta)$val) - mean(ufunc(z[sub], xi, theta)$val)
  
  
  uat2.diff[i] <- ufunc(-2, xi, theta)$val - ufunc(2, xi, theta)$val  # Record difference of energies at x=-2 and x=2
  
  ## Plot
  hist(z, 100, xlim=c(-4,4), main= c(mean(z>0), ar.ave[i]) )
  
  plot(ax, ufunc(ax, xi, theta)$val, type="l", xlim=c(-6,6), main=i)
  lines(ax, ax^2/10^2, lty=2)
  abline(v=c(-4,-2,2,4), lty=3)
  
  gr <- apply(gradfunc(x[sub], xi), 2, mean) - apply(gradfunc(z[sub], xi), 2, mean)
  plot(xi, gr)
  
  lr <- scheduler(lr, i, milestones, adj=.2)
}


par(mfrow=c(2,3))
hist(z, 100, xlim=c(-4,4), main= mean(z>0))

plot( ax, ufunc(ax, xi, theta)$val, type="l", xlim=c(-6,6), main=i)
lines( ax, ax^2 /10^2, lty=2 )
abline(v=c(-4,-2,2,4), lty=3)

gr <- apply( gradfunc(x[sub],xi),2,mean) - apply( gradfunc(z[sub],xi), 2, mean)
plot(xi, gr, ylim=c(-.2,.2))
abline(v=0, lty=2)

hist(x, 100, xlim=c(-4,4))

ax <- seq(-5, 5, .025)
plot(ax, ufunc0(ax), type="l", ylim=c(0,5))
plot(ar.ave)

hist(z[x<0],100)
hist(z[x>0],100)
hist(z,100)

plot(st.dif, type="l", main="loss")
abline(h=0)

plot(1:niter, uat2.diff, type="l")
abline(h=ufunc0(-2) - ufunc0(2), lty=2)

plot(theta0, theta)
abline(c(0,1), lty=2)

# Long-run Sampling
out.long <- langev(x, 1000, langev.eps, theta)
hist(out.long$z,100, main="long", xlab="x")

# Post-training Sampling
out.post.sd1 <- langev(rnorm(1000, sd=1), 1000, langev.eps, theta)
hist(out.post.sd1$z,100, main="post (start from Normal sd=1)", xlab="x")

par(mfrow=c(2,2))

plot(ax, ufunc(ax, xi, theta)$val - min(ufunc(ax, xi, theta)$val), type="l", xlim=c(-3,3), ylim=c(0, 50),
      main="Energy Function", ylab="Energy", xlab="x", col=rgb(31/255, 119/255, 180/255), lwd=2)
abline(v=c(-4,-2,2,4), lty=3)
lines(ax, ufunc0(ax) - min( ufunc0(ax)), col=rgb(1,127/255,14/255), lwd=2)
legend("topright", legend=c("Truth", "Learned"),
       col=c(rgb(1,127/255,14/255), rgb(31/255, 119/255, 180/255)), lty=1, lwd=2)

ufunc0_exp <- function(x) {
  alp *exp(-(x+b)^2/2/sig^2)*1/sqrt(2*pi) + (1-alp) *exp(-(x-b)^2/2/sig^2 )*1/sqrt(2*pi)  
}

h <- hist(z, 100, plot=F)
h$counts <- h$counts / sum(h$counts)
plot(h, freq=TRUE, ylab="Relative Frequency",  main="Replay Buffer", xlab="x", col=rgb(31/255, 119/255, 180/255), ylim=c(0,.3), xlim=c(-3,3))
lines(ax, ufunc0_exp(ax), col=rgb(1,127/255,14/255), lwd=2)
axis(side=2,at=seq(0,.3,.05),labels=seq(0,.3,.05))
legend("topright", legend=c("Truth"),
       col=rgb(1,127/255,14/255), lty=1, lwd=2)


h <- hist(out.long$z, 100, plot=F)
h$counts <- h$counts / sum(h$counts)
plot(h, freq=TRUE, ylab="Relative Frequency",  main="Long-run Samples", xlab="x", col=rgb(31/255, 119/255, 180/255), ylim=c(0,.3), xlim=c(-3,3))
lines(ax, ufunc0_exp(ax), col=rgb(1,127/255,14/255), lwd=2)
axis(side=2,at=seq(0,.3,.05),labels=seq(0,.3,.05))
legend("topright", legend=c("Truth"),
       col=rgb(1,127/255,14/255), lty=1, lwd=2)

h <- hist(out.post.sd1$z, 100, plot=F)
h$counts <- h$counts / sum(h$counts)
plot(h, freq=TRUE, ylab="Relative Frequency",  main="Post-training Samples", xlab="x", col=rgb(31/255, 119/255, 180/255), ylim=c(0,.3), xlim=c(-3,3))
lines(ax, ufunc0_exp(ax), col=rgb(1,127/255,14/255), lwd=2)
axis(side=2,at=seq(0,.3,.05),labels=seq(0,.3,.05))
legend("topright", legend=c("Truth"),
       col=rgb(1,127/255,14/255), lty=1, lwd=2)

