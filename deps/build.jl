target = "Garbo/libgarbo.$(Libdl.dlext)"
vers = "0.1"

if !isfile(target)
    #if is_linux()
        LibGit2.clone("https://github.com/kpamnany/Garbo", "Garbo")
        println("Compiling libgarbo...")
        run(`make -C Garbo`)
    #else
#	error("Garbo is Linux-only right now")
#    end
end

