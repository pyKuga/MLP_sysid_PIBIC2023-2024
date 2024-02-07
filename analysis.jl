#School of Mechanical Engineering - State University of Campinas
#Paulo Yoshio Kuga

#This package is beign written for "Análise de Robustez em Métodos de Identificação de Sistemas", a undergraduate ressearch for PIBIC.

#Main content: automated sys analysis verifications

#gets the amplitude and the phase angle of a signal shifted fourier transform 
function FourierTransformAnalysis(Y)
    F = fft(Y) |> fftshift;
    #freqs = fftfreq(ns, fs) |> fftshift;
    A = abs.(F)
    θ = atan.(imag(F),real(F));
    return A,θ#,freqs
end    

function FourierTransformAnalysis(Y,ns,fs)
    F = fft(Y) |> fftshift;
    freqs = fftfreq(ns, fs) |> fftshift;
    A = abs.(F)
    θ = atan.(imag(F),real(F));
    return A,θ,freqs
end    


#this is a non-linear case, where the signal is imputed. when the signal is non-linear, 
#it's senseless to apply a bode transform, since we don't know the poles of the system.
function runAnalysis(sysT,Yor, U)
    
    Ysim,_,_= lsim(sysT,U,t);

    mOr, pOr = FourierTransformAnalysis(Yor); 
    msim, psim = FourierTransformAnalysis(Ysim); 


    timeNorm = norm(Ysim-Yor)/norm(Yor);
    magNorm = norm(msim-mOr)/norm(mOr);
    phaseNorm = norm(psim-pOr)/norm(pOr);

    triple = [timeNorm; magNorm; phaseNorm]

    return triple

    return timeNorm

end

function runAnalysis(sysT,Yor)
    Ysim,_,_= impulse(sysT,t);
    timeNorm = norm(Ysim-Yor)/norm(Yor);
    return timeNorm

end


function runAnalysis(sysT,Yor, U,mOr, pOr,t)
    Ysim,_,_= lsim(sysT,U,t);
    msim, psim, = bode(sysT,w);

    timeNorm = norm(Ysim-Yor)/norm(Yor);
    magNorm = norm(msim-mOr)/norm(mOr);
    phaseNorm = norm(psim-pOr)/norm(pOr);

    triple = [timeNorm; magNorm; phaseNorm]

    return triple

end

function n4sidPatternArg(Yout,U,dt,nx)
    Data = iddata(Yout,U,dt);
    sys_n4sid = n4sid(Data, nx,zeroD=true);
    return sys_n4sid
end





# _,_,p = damp(sysD);
# _,_,p4 = damp(sys_n4sid);
# _,_,pe = damp(sys_era);
# _,_,pa = damp(sys_arx);


# v_era = metrica(p,pe)
# v_n4sid = metrica(p,p4)
# v_arx = metrica(p,pa)




#sys_arx = arx(Data,2,4,stochastic=false) #arx é MISO



#Yi,Ti,Xi = lsim(sys_n4sid,U',t,x0=X0);

#plot(t,[X' Xi'],layout=4)

#processar o ruído
#avaliar a sensibilidade ao ruído


#FFTf1 = abs.(fft(U));
#fw = 500
#plot(FFTf1[1:fw])



