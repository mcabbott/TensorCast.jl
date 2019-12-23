function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(Test.do_test),Test.Returned,Expr})
    precompile(Tuple{typeof(Test.eval_test),Expr,Expr,LineNumberNode,Bool})
    precompile(Tuple{typeof(Test.finish),Test.DefaultTestSet})
    precompile(Tuple{typeof(Test.get_alignment),Test.DefaultTestSet,Int64})
    precompile(Tuple{typeof(Test.get_test_result),Expr,LineNumberNode})
    precompile(Tuple{typeof(Test.print_counts),Test.DefaultTestSet,Int64,Int64,Int64,Int64,Int64,Int64,Int64})
    precompile(Tuple{typeof(Test.testset_beginend),Tuple{String,Expr},Expr,LineNumberNode})
end
