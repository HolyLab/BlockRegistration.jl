module RegisterGUI

using Images, Gtk, Colors, Graphics, MappedArrays
using RegisterCore, CenterIndexedArrays
import ImagePlayer
using ImagePlayer.imshow

# export selectbb3, showoverlay, displayblocks, displaymismatch
export showoverlay, displaymismatch

# # Tk's Messagebox blocks all interaction until dismissed; this allows you to
# # interact with graphics elements and then click a button when finished
# function modalmessage(message::String, buttons::String...; title::String = "")
#     win = Toplevel(title)
#     f = Frame(win, padding = [3,3,2,2], relief="groove")
#     pack(f, expand = true, fill = "both")
#     txt = Text(f)
#     set_value(txt, message)
#     grid(txt, 1, 1, sticky="nsew")
#     grid_columnconfigure(f, 1, weight=1)
#     grid_rowconfigure(f, 1, weight=1)
#     fbuttons = Frame(f)
#     grid(fbuttons, 2, 1)
#     bhandles = [Button(fbuttons, b) for b in buttons]
#     c = Condition()
#     for i = 1:length(bhandles)
#         grid(bhandles[i], 1, i)
#         bind(bhandles[i], "command", path->notify(c, buttons[i]))
#     end
#     set_size(win, 300, 200)
#     Tk.update()
#     val = wait(c)
#     destroy(win)
#     val
# end

# function selectbb3(img; basestack = (size(img, "t")+1)>>1)
#     dat = sliceim(img, "t", basestack)
#     A = dat[:,:,:]
#     so = spatialorder(dat)
#     p3 = squeeze(max(A, (), 3), 3)
#     dat3 = sliceim(dat, :, :, 1) # Just to generate the correct properties
#     imsl3 = share(dat3, p3)
#     imgc3, img3 = view(share(dat3, p3), xy=so[1:2], name="xy selection", clim=(100,2500))
#     p1 = squeeze(max(A, (), 1), 1)
#     dat1 = sliceim(dat, 1, :, :)
#     imgc1, img1 = view(share(dat1, p1), xy=so[2:3], name="z selection", clim=(100,2500))
#     choice = modalmessage("Click OK when you've set the zoom region in both plots", "OK", "Cancel", title="Finish")
#     ret = nothing
#     if choice=="OK"
#         index1 = round(Int, xmin(img3.zoombb)):round(Int, xmax(img3.zoombb))
#         index2 = round(Int, ymin(img3.zoombb)):round(Int, ymax(img3.zoombb))
#         index3 = round(Int, ymin(img1.zoombb)):round(Int, ymax(img1.zoombb))
#         ret = index1, index2, index3, basestack
#     end
#     destroy(toplevel(imgc3))
#     destroy(toplevel(imgc1))
#     return ret
# end

function showoverlay(img1, img2; clim = (0,1), kwargs...)
    if length(clim) != 2
        error("clim must be either (min,max) or ((min1,max1),(min2,max2))")
    end
    if isa(clim[1], Number)
        clim = (clim,clim)
    end
    imgsc = scaleimages((img1, img2), clim)
    ovr = colorview(RGB, imgsc[1], imgsc[2], zeroarray)
    imshow(ovr; kwargs...)
end

function showoverlay(img1, img2, img3; clim = (0,1), kwargs...)
    if length(clim) != 2 && length(clim) != 3
        error("clim must be either (min,max) or ((min1,max1),(min2,max2),(min3,max3))")
    end
    if isa(clim[1], Number)
        clim = (clim,clim,clim)
    end
    imgsc = scaleimages((img1, img2, img3), clim)
    ovr = colorview(RGB, imgsc[1], imgsc[2], imgsc[3])
    imshow(ovr; kwargs...)
end

function scaleimages(imgs, clims)
    map((img,cl)->mappedarray(x->clamp01nan(scaleminmax(cl...)(x)), img), imgs, clims)
end

# function displayblocks(gridsize, maxshift, fixed, moving = nothing; blocksize = RegisterMismatch.defaultblocksize(fixed, gridsize), clim = nothing)
#     assert2d(fixed)
#     assert2d(moving)
#     dfixed = isa(data(fixed), SubArray) ? mycopy!(similar(data(fixed)), data(fixed)) : data(fixed)
#     canvases = canvasgrid(gridsize[1], gridsize[2], pad=5, name="image blocks")
#     padsz = RegisterMismatch.padsize(blocksize, maxshift)
#     padszt = tuple(padsz...)
#     getindexes = RegisterMismatch.padranges(blocksize, maxshift)
#     padded1 = zeros(padszt)
#     padded2 = zeros(padszt)
#     N = length(gridsize)
#     blk = [b>>1 for b in blocksize]
#     fixedsub = sub(padded1, ntuple(ndims(fixed), d->(1:(2blk[d]+1))+maxshift[d]))
#     center = Array(Int, N)
#     for c in Counter(gridsize)
#         fill!(padded1, 0)
#         fill!(padded2, 0)
#         for i = 1:N
#             center[i] = round(Int, (size(fixed, i)-1)*(c[i]-1)/(gridsize[i]-1)+1)
#         end
#         rng = ntuple(N, i->(-blk[i]:blk[i])+center[i])
#         myget!(fixedsub, dfixed, rng, 0.0)
#         if moving != nothing
#             if clim == nothing
#                 error("Must specify clim")
#             end
#             msnip = sub(moving, rng...)
#             myget!(padded2, msnip, tuple(getindexes...), 0.0)
#             ovr = Overlay((copy(padded1),copy(padded2)), (RGB(1,0,1),RGB(0,1,0)), (clim,clim))
#             view(canvases[c...], ovr, pixelspacing=pixelspacing(moving))
#         else
#             if clim != nothing
#                 view(canvases[c...], padded1', pixelspacing=pixelspacing(moving), clim=clim)
#             else
#                 view(canvases[c...], padded1', pixelspacing=pixelspacing(moving))
#             end
#         end
#     end
# end

function displaymismatch(mms; thresh = 0, totaldenom::Bool = false, clim=:shared, umin = nothing, ucmp = nothing)
    gridsize = size(mms)
    nd = length(gridsize)
    nd == 2 || error("Cannot display grid of $nd dimensions")
    nblocks = prod(gridsize)
    cgrid = canvasgrid(gridsize)
    local D
    if totaldenom
        D = zeros(size(first(mms)))
        for j = 1:gridsize[2], i = 1:gridsize[1]
            _, denom = separate(mms[i,j])
            D += denom
        end
        D /= nblocks
        function ratioD(mm)
            num, _ = separate(mm)
            r = num./D
            r[D .< thresh] = NaN
            r
        end
        rs = map(mm->ratioD(mm), mms)
    else
        rs = map(mm->ratio(mm, thresh), mms)
    end
    local mx
    if clim == :shared
        mx = mapreduce(maxfinite, max, zero(eltype(first(rs))), rs)
    end
    for j = 1:gridsize[2], i = 1:gridsize[1]
        r = rs[i,j]
        if clim == :shared
            cl = (0,mx)
        elseif clim == :separate
            cl = (0,maxfinite(r))
        else
            cl = clim
        end
        imshow(cgrid[i,j], r.data, clim=cl)#, pixelspacing=[1,1])
        # if umin != nothing
        #     tmpy,tmpx = getyx(umin,i,center)
        # else
        #     ind = indminmismatch(num, denom, thresh)
        #     tmpy,tmpx = ind2sub(size(r), ind)
        # end
        # annotate!(imgc, img2, AnnotationText(tmpx, tmpy, "x", color=RGB(0,1,0)))
        # if ucmp != nothing
        #     tmpy,tmpx = getyx(ucmp,i,center)
        #     annotate!(imgc, img2, AnnotationText(tmpx, tmpy, "x", color=RGB(1,0,1)))
        # end
        # push!(imgclist, imgc)
        # push!(img2list, img2)
    end
    nothing
end
function displaymismatch(mm::CenterIndexedArray; kwargs...)
    mms = Array(Any, 1, 1)
    mms[] = mm
    displaymismatch(mms, kwargs...)
end

# getyx{T<:FloatingPoint}(u::Array{T}, i, center) = u[1,i]+center[1], u[2,i]+center[2]
# getyx(u, i, center) = u[i] + center

# function marku(u::Array, maxshift, imgclist, img2list)
#     if size(u,1) != 2
#         error("Works only in 2d")
#     end
#     ux = squeeze(u[2,:,:,1], 1)
#     uy = squeeze(u[1,:,:,1], 1)
#     cx = ux .+ (maxshift[2] + 1)
#     cy = uy .+ (maxshift[1] + 1)
#     for i = 1:length(imgclist)
#         annotate!(imgclist[i], img2list[i], AnnotationText(cx[i], cy[i], "x", color=RGB(0,0,1)))
#     end
# end

end
