#include "duckdb/core_functions/aggregate/algebraic_functions.hpp"
#include "duckdb/core_functions/aggregate/sum_helpers.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

template <class T>
struct AvgState {
	uint64_t count;
	T value;

	void Initialize() {
		this->count = 0;
	}

	void Combine(const AvgState<T> &other) {
		this->count += other.count;
		this->value += other.value;
	}
};

struct KahanAvgState {
	uint64_t count;
	double value;
	double err;

	void Initialize() {
		this->count = 0;
		this->err = 0.0;
	}

	void Combine(const KahanAvgState &other) {
		this->count += other.count;
		KahanAddInternal(other.value, this->value, this->err);
		KahanAddInternal(other.err, this->value, this->err);
	}
};

struct AverageDecimalBindData : public FunctionData {
	explicit AverageDecimalBindData(double scale) : scale(scale) {
	}

	double scale;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AverageDecimalBindData>(scale);
	};

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<AverageDecimalBindData>();
		return scale == other.scale;
	}
};

struct AverageSetOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.Initialize();
	}
	template <class STATE>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.Combine(source);
	}
	template <class STATE>
	static void AddValues(STATE &state, idx_t count) {
		state.count += count;
	}
};

template <class T>
static T GetAverageDivident(uint64_t count, optional_ptr<FunctionData> bind_data) {
	T divident = T(count);
	if (bind_data) {
		auto &avg_bind_data = bind_data->Cast<AverageDecimalBindData>();
		divident *= avg_bind_data.scale;
	}
	return divident;
}

struct IntegerAverageOperation : public BaseSumOperation<AverageSetOperation, RegularAdd> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (state.count == 0) {
			finalize_data.ReturnNull();
		} else {
			double divident = GetAverageDivident<double>(state.count, finalize_data.input.bind_data);
			target = double(state.value) / divident;
		}
	}
};

struct IntegerAverageOperationHugeint : public BaseSumOperation<AverageSetOperation, AddToHugeint> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (state.count == 0) {
			finalize_data.ReturnNull();
		} else {
			long double divident = GetAverageDivident<long double>(state.count, finalize_data.input.bind_data);
			target = T(Hugeint::Cast<long double>(state.value) / divident);
		}
	}
};

// Averaging DATE is like averaging integers to produce a fixed point number (TIMESTAMP)
struct IntegerAverageOperationDate : public BaseSumOperation<AverageSetOperation, AddToHugeint> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (state.count == 0) {
			finalize_data.ReturnNull();
		} else {
			const auto divisor = GetAverageDivident<long double>(state.count, finalize_data.input.bind_data);
			const auto dividend = Hugeint::Cast<long double>(state.value) * Interval::MICROS_PER_DAY;
			target = T(dividend / divisor);
		}
	}
};

// Averaging TIMETZ is really a matter for theologians,
// so we just average the time components to produce a TIME.
struct AddToTimeTZ {
	template <class STATE, class T>
	static void AddNumber(STATE &state, T input) {
		AddToHugeint::AddNumber(state, dtime_tz_t::decode_micros(input));
	}
	template <class STATE, class T>
	static void AddConstant(STATE &state, T input, idx_t count) {
		AddToHugeint::AddConstant(state, dtime_tz_t::decode_micros(input), count);
	}
};

struct IntegerAverageOperationTimeTZ : public BaseSumOperation<AverageSetOperation, AddToTimeTZ> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (state.count == 0) {
			finalize_data.ReturnNull();
		} else {
			const auto divisor = GetAverageDivident<long double>(state.count, finalize_data.input.bind_data);
			const auto dividend = Hugeint::Cast<long double>(state.value);
			target = T(dividend / divisor);
		}
	}
};

struct HugeintAverageOperation : public BaseSumOperation<AverageSetOperation, HugeintAdd> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (state.count == 0) {
			finalize_data.ReturnNull();
		} else {
			long double divident = GetAverageDivident<long double>(state.count, finalize_data.input.bind_data);
			target = T(Hugeint::Cast<long double>(state.value) / divident);
		}
	}
};

struct NumericAverageOperation : public BaseSumOperation<AverageSetOperation, RegularAdd> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (state.count == 0) {
			finalize_data.ReturnNull();
		} else {
			target = state.value / state.count;
		}
	}
};

struct KahanAverageOperation : public BaseSumOperation<AverageSetOperation, KahanAdd> {
	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (state.count == 0) {
			finalize_data.ReturnNull();
		} else {
			target = (state.value / state.count) + (state.err / state.count);
		}
	}
};

AggregateFunction GetAverageAggregate(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT8: {
		return AggregateFunction::UnaryAggregate<AvgState<int64_t>, int8_t, double, IntegerAverageOperation>(
		    LogicalType::TINYINT, LogicalType::DOUBLE);
	}
	case PhysicalType::INT16: {
		return AggregateFunction::UnaryAggregate<AvgState<int64_t>, int16_t, double, IntegerAverageOperation>(
		    LogicalType::SMALLINT, LogicalType::DOUBLE);
	}
	case PhysicalType::INT32: {
		return AggregateFunction::UnaryAggregate<AvgState<hugeint_t>, int32_t, double, IntegerAverageOperationHugeint>(
		    LogicalType::INTEGER, LogicalType::DOUBLE);
	}
	case PhysicalType::INT64: {
		return AggregateFunction::UnaryAggregate<AvgState<hugeint_t>, int64_t, double, IntegerAverageOperationHugeint>(
		    LogicalType::BIGINT, LogicalType::DOUBLE);
	}
	case PhysicalType::INT128: {
		return AggregateFunction::UnaryAggregate<AvgState<hugeint_t>, hugeint_t, double, HugeintAverageOperation>(
		    LogicalType::HUGEINT, LogicalType::DOUBLE);
	}
	default:
		throw InternalException("Unimplemented average aggregate");
	}
}

AggregateFunction GetAverageAggregate(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::DECIMAL:
		return GetAverageAggregate(type.InternalType());
		break;
	case LogicalTypeId::DATE:
		return AggregateFunction::UnaryAggregate<AvgState<hugeint_t>, int32_t, int64_t, IntegerAverageOperationDate>(
		    type, LogicalTypeId::TIMESTAMP);
	case LogicalTypeId::TIME_TZ:
		return AggregateFunction::UnaryAggregate<AvgState<hugeint_t>, uint64_t, int64_t, IntegerAverageOperationTimeTZ>(
		    type, LogicalTypeId::TIME);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return AggregateFunction::UnaryAggregate<AvgState<hugeint_t>, int64_t, int64_t, IntegerAverageOperationHugeint>(
		    type, type);
	default:
		throw InternalException("Unimplemented average aggregate");
	}
}

unique_ptr<FunctionData> BindDecimalAvg(ClientContext &context, AggregateFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	auto decimal_type = arguments[0]->return_type;
	function = GetAverageAggregate(decimal_type);
	function.name = "avg";
	function.arguments[0] = decimal_type;
	function.return_type = LogicalType::DOUBLE;
	return make_uniq<AverageDecimalBindData>(
	    Hugeint::Cast<double>(Hugeint::POWERS_OF_TEN[DecimalType::GetScale(decimal_type)]));
}

AggregateFunctionSet AvgFun::GetFunctions() {
	AggregateFunctionSet avg;

	avg.AddFunction(AggregateFunction({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr,
	                                  nullptr, nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr,
	                                  BindDecimalAvg));
	avg.AddFunction(GetAverageAggregate(LogicalType::TINYINT));
	avg.AddFunction(GetAverageAggregate(LogicalType::SMALLINT));
	avg.AddFunction(GetAverageAggregate(LogicalType::INTEGER));
	avg.AddFunction(GetAverageAggregate(LogicalType::BIGINT));
	avg.AddFunction(GetAverageAggregate(LogicalType::HUGEINT));
	avg.AddFunction(AggregateFunction::UnaryAggregate<AvgState<double>, double, double, NumericAverageOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE));

	avg.AddFunction(GetAverageAggregate(LogicalType::DATE));
	avg.AddFunction(GetAverageAggregate(LogicalType::TIME));
	avg.AddFunction(GetAverageAggregate(LogicalType::TIME_TZ));
	avg.AddFunction(GetAverageAggregate(LogicalType::TIMESTAMP));
	avg.AddFunction(GetAverageAggregate(LogicalType::TIMESTAMP_TZ));
	return avg;
}

AggregateFunction FAvgFun::GetFunction() {
	return AggregateFunction::UnaryAggregate<KahanAvgState, double, double, KahanAverageOperation>(LogicalType::DOUBLE,
	                                                                                               LogicalType::DOUBLE);
}

} // namespace duckdb
