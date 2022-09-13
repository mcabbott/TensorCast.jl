""" These tests are inspired by the paper published on python's "einops" package.

See 'Alex Rogozhnikov, 2022. "EINOPS: CLEAR AND RELIABLE TENSOR MANIPULATIONS WITH EINSTEIN-LIKE NOTATION",
published at ICLR 2022'; Listing 1.
https://openreview.net/pdf?id=oapKSVM2bcj
"""

@testset "Einops paper - Listing 1" begin

  # Format:
  # NUMBER NUMPY-CODE
  #        EINOPS-CODE
  # JULIA-CODE...

  # 1 np.transpose(x, [0, 3, 1, 2]) 
  #   rearrange(x, 'b h w c -> b c h w')
  @testset "Test 1" begin
    x = rand(3, 4, 5, 6)
    @test (
      permutedims(x, [1, 4, 2, 3]) 
      == @cast _[b, c, h, w] := x[b, h, w, c])
  end

  # 2 np.reshape(x, [x.shape[0]*x.shape[1], x.shape[2]]) 
  #   rearrange(x, 'h w c -> (h w) c ') 
  @testset "Test 2" begin
    x = rand(3, 4, 5)
    @test (
      reshape(x, size(x, 1) * size(x, 2), size(x, 3)) == 
      @cast _[(h, w), c] := x[h, w, c]
    )
  end

  # 3 np.squeeze(x, 0) 
  #   rearrange(x, '() h w c -> h w c') 
  @testset "Test 3" begin
    x = rand(1, 2, 3, 4)
    @test (
      reshape(x, 2, 3, 4) ==
      @cast _[h, w, c] := x[_, h, w, c])
  end

  # 4 np.expand_dims(x, -1) 
  #   rearrange(x, 'h w c -> h w c ()') 
  @testset "Test 4" begin
    x = rand(2, 3, 4)
    @test (
      reshape(x, 2, 3, 4, 1) ==
      @cast _[h, w, c, 1] := x[h, w, c])
  end

  # 5 np.stack([r, g, b], axis=2) 
  #   rearrange([r, g, b], 'c h w -> h w c') 
  @testset "Test 5" begin
    xs = [rand(2, 2) for _ in 1:3]
    @test (
      cat(xs..., dims=3) ==
      @cast _[h, w, i] := xs[i][h, w])
  end

  # 6 np.concatenate([r, g, b], axis=0) 
  #   rearrange([r, g, b], 'c h w -> (c h) w')
  @testset "Test 6" begin
    xs = [rand(2, 2) for _ in 1:3]
    expand_first(x) = reshape(x, 1, size(x)...)
    @test (
      cat(expand_first.(xs)..., dims=1) ==
      @cast _[i, h, w] := xs[i][h, w]
    )
  end

  # 7 np.flatten(x) 
  #   rearrange(x, 'b t c -> (b t c) ') 
  @testset "Test 7" begin
    x = rand(3, 4, 5)
    @test (
      x[:] == 
      @cast _[(b, t, c)] := x[b, t, c]
    )
  end

  # 8 np.swap_axes(x, 0, 1) 
  #   rearrange(x, 'b t c -> t b c') 
  @testset "Test 8" begin
    x = rand(3, 4, 5)
    @test (
      permutedims(x, (2, 1, 3)) == 
      @cast _[t, b, c] := x[b, t, c]
    )
  end

  # 9 left, right = np.split(image, 2, axis=1) 
  #   rearrange(x, 'h (lr w) c -> lr h w c', lr=2) 
  @testset "Test 9" begin
    image = rand(2, 32, 32, 3)
    @test (
      hcat(image[1, :, :, :], image[2, :, :, :]) == 
      @cast _[h, (w, lr), c] := image[lr, h, w, c] (lr in 1:2)
    )
  end

  # 10 even, odd = x[:, 0::2], x[:, 1::2] 
  #    rearrange(x, 'h (w par) -> par h w c', par=2) 
  @testset "Test 10" begin
    x = rand(7, 10)
    @test (
      [x[:, 1:2:end], x[:, 2:2:end]] == 
      @cast _[par][h, w] := x[h, (par, w)] (par in 1:2)
    )
  end

  # 11 np.max(x, [1, 2]) 
  #    reduce(x, 'b h w c -> b c', 'max') 
  @testset "Test 11" begin
    x = rand(2, 2, 3, 4)
    @test (
      maximum(x, dims=[2, 3])[:, 1, 1, :] == 
      @reduce _[b, c] := maximum(h, w) x[b, h, w, c]
    )
  end

  # 12 np.mean(x) 
  #    reduce(x, 'b h w c ->', 'mean') 
  @testset "Test 12" begin
    x = rand(2, 2, 3, 4)
    @test (
      [mean(x)] == 
      @reduce _[_] := mean(b, h, w, c) x[b, h, w, c]
    )
  end

  # 13 np.mean(x, axis=(1, 2), keepdims=True) 
  #    reduce(x, 'b h w c -> b () () c', 'mean') 
  @testset "Test 13" begin
    x = rand(2, 2, 3, 4)
    @test (
      mean(x, dims=[2, 3]) == 
      @reduce _[b, _, _, c] := mean(h, w) x[b, h, w, c]
    )
  end

  # 14 np.reshape(x, [-1, 2]).max(axis=1) 
  #    reduce(x, '(h 2) -> h', 'max') 
  @testset "Test 14" begin
    x = rand(5, 4, 3, 2)
    @test (
      maximum(reshape(x, :, 2), dims=[2])[:] == 
      @reduce _[(h, w, c)] := maximum(l) x[h, w, c, l] (l in 1:2)
    )
  end

  # 15 np.repeat(x, 2, axis=1) 
  #    repeat(x, 'h w -> h (w 2)') 
  @testset "Test 15" begin
    x = rand(3, 4)
    @test (
      repeat(x; inner=(1, 2)) ==
      @cast _[h, (rep, w)] :=  x[h, w] (rep in 1:2)
    )
  end

  # 16 np.tile(x, 2, axis=1) 
  #    repeat(x, 'h w -> h (2 w)') 
  @testset "Test 16" begin
    x = rand(3, 4)
    @test (
      repeat(x; outer=(1, 2)) ==
      @cast _[h, (w, rep)] :=  x[h, w] (rep in 1:2)
    )
  end

  # 17 np.tile(x[:, :, np.newaxis], 3, axis=2) 
  #    repeat(x, 'h w -> h w 3')
  @testset "Test 17" begin
    x = rand(3, 4)
    @test (
      repeat(x, 1, 1, 3) ==
      @cast _[h, w, rep] :=  x[h, w] (rep in 1:3)
    )
  end

end
