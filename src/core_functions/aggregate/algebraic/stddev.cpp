#include "duckdb/core_functions/aggregate/algebraic_functions.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/core_functions/aggregate/algebraic/stddev.hpp"
#include <cmath>

namespace duckdb {

AggregateFunctionSet StdDevSampFun::GetFunctions() {
	AggregateFunctionSet funcs;
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, double, double, STDDevSampOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE));

	// Temporal types all return INTERVAL for their STDDEV
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, date_t, interval_t, STDDevSampOperation>(
	    LogicalType::DATE, LogicalType::INTERVAL));
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, dtime_t, interval_t, STDDevSampOperation>(
	    LogicalType::TIME, LogicalType::INTERVAL));
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, dtime_tz_t, interval_t, STDDevSampOperation>(
	    LogicalType::TIME_TZ, LogicalType::INTERVAL));
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, timestamp_t, interval_t, STDDevSampOperation>(
	    LogicalType::TIMESTAMP, LogicalType::INTERVAL));
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, timestamp_t, interval_t, STDDevSampOperation>(
	    LogicalType::TIMESTAMP_TZ, LogicalType::INTERVAL));
	return funcs;
}

AggregateFunctionSet StdDevPopFun::GetFunctions() {
	AggregateFunctionSet funcs;
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, double, double, STDDevPopOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE));

	// Temporal types all return INTERVAL for their STDDEV
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, date_t, interval_t, STDDevPopOperation>(
	    LogicalType::DATE, LogicalType::INTERVAL));
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, dtime_t, interval_t, STDDevPopOperation>(
	    LogicalType::TIME, LogicalType::INTERVAL));
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, dtime_tz_t, interval_t, STDDevPopOperation>(
	    LogicalType::TIME_TZ, LogicalType::INTERVAL));
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, timestamp_t, interval_t, STDDevPopOperation>(
	    LogicalType::TIMESTAMP, LogicalType::INTERVAL));
	funcs.AddFunction(AggregateFunction::UnaryAggregate<StddevState, timestamp_t, interval_t, STDDevPopOperation>(
	    LogicalType::TIMESTAMP_TZ, LogicalType::INTERVAL));
	return funcs;
}

AggregateFunction VarPopFun::GetFunction() {
	return AggregateFunction::UnaryAggregate<StddevState, double, double, VarPopOperation>(LogicalType::DOUBLE,
	                                                                                       LogicalType::DOUBLE);
}

AggregateFunction VarSampFun::GetFunction() {
	return AggregateFunction::UnaryAggregate<StddevState, double, double, VarSampOperation>(LogicalType::DOUBLE,
	                                                                                        LogicalType::DOUBLE);
}

AggregateFunction StandardErrorOfTheMeanFun::GetFunction() {
	return AggregateFunction::UnaryAggregate<StddevState, double, double, StandardErrorOfTheMeanOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE);
}

} // namespace duckdb
