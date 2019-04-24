import Denoising_GAN

N = 16384
gan = GAN(N)
gan.generator2()
gan.train(86,100,20,11572)
