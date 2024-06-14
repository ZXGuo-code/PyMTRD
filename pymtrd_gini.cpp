#include<iostream>
#include<algorithm>
#define DELTA 0.000001
using namespace std;

extern "C"
{
	//g++ -o pymtrd_gini.so -shared -fPIC pymtrd_gini.cpp
	//Get the number of days in the year according to the input year
	int get_days_year(int year)
	{
		if (((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0))
		{
			return 366;
		}
		else
		{
			return 365;
		}
	}


	//Get the max value of the input array
	float get_array_max(float* input, int length)
	{
		float maximum = input[0];
		for (int ii = 0; ii < length; ii++)
		{
			if (input[ii] > maximum)
			{
				maximum = input[ii];
			}
		}
		return maximum;
	}


	//Get the min value of the input array
	float get_array_min(float* input, int length)
	{
		float minimum = input[0];
		for (int ii = 0; ii < length; ii++)
		{
			if (input[ii] < minimum)
			{
				minimum = input[ii];
			}
		}
		return minimum;
	}


	//Calculate the cumulative value of the input array
	void cumsum(float* input, float* output, int length)
	{
		output[0] = input[0];
		for (int ii = 1; ii < length; ii++)
		{
			output[ii] = output[ii - 1] + input[ii];
		}
	}


	//Trapezoidal method of integration
	float get_trapz(float* yarry, float* xarry, int length)
	{
		float area = 0.0;
		for (int ii = 0; ii < length - 1; ii++)
		{
			float y_ave = yarry[ii] + yarry[ii + 1];
			y_ave *= 0.5;
			float step_x = xarry[ii + 1] - xarry[ii];
			area += y_ave * step_x;
		}
		return area;
	}


	//Extend the input time series
	float* extend_ts(float* input, int nyears, int start_year, int length)
	{
		int extend_days = 366;
		if((nyears == 1)&&(get_days_year(start_year)==365))
		{
			extend_days = 365;
		}
		//final year's data supplemented by first year's data
		int days_all = length+extend_days;
		float* output = new float[days_all];
		//original data
		for(int ii=0; ii<length; ii++)
		{
			output[ii] = input[ii];
		}
		//supplementary data
		for(int ii=0; ii<extend_days; ii++)
		{
			output[ii+length]=input[ii];
		}
		return output;
	}


	//Add and remove elements to a sorted array, which is used to avoid repeated sorting
	int add_remove(float* input, int length, float remove, float add, float threshold, float nodata_value)
	{
		int mark_add = 0;
		int mark_remove = 0;
		//remove element
		if(remove < threshold || abs(remove-nodata_value)<DELTA){}
		else
		{
			//get the location of the data to be removed
			float difference = abs(remove-input[0]);
			mark_remove = 0;
			for(int ii = 1; ii<length; ii++)
			{
				//search by order
				if(abs(remove-input[ii])<difference)
				{
					mark_remove = ii;
					difference=abs(remove-input[ii]);
				}
			}
			//remove data
			length--;
			for(int ii = mark_remove; ii < length; ii++)
			{
				input[ii] = input[ii+1];
			}
		}
		//add element
		if(add < threshold || abs(add-nodata_value)<DELTA){}
		else
		{
			length++;
			mark_add=length-1;
			//get the location where the data needs to be inserted
			for(int ii = 0; ii<length-1; ii++)
			{
				if(add-input[ii]<0.0)
				{
					mark_add = ii;
					break;
				}
			}
			//insert data
			for(int ii = length-1; ii>mark_add;ii--)
			{
				input[ii] = input[ii-1];
			}
			input[mark_add] = add;
		}
		return length;
	}


	//Add and remove elements to a array
	int add_remove_ugini(float* input, int length, float remove, float add, float nodata_value)
	{
		if(remove<-1 || abs(remove-nodata_value)<DELTA){}
		else
		{
			//remove data
			length--;
			for(int ii=0; ii<length; ii++)
			{
				input[ii] = input[ii+1];
			}
		}
		if(add<-1 || abs(add-nodata_value)<DELTA){}
		else
		{
			//insert data
			length++;
			input[length-1]=add;
		}
		return length;
	}


	//Get the unranked Gini index of the input daily-scale time series rainfall data
	float get_ugini(float* ts_daily, int length, float nodata_value)
	{
		float ugini = nodata_value;
		//calculate the cumulative value
		float* cum_wealths_u = new float[length];
		cumsum(ts_daily, cum_wealths_u, length);
		float sum_wealths_u = cum_wealths_u[length - 1];
		//calculate the unranked Gini index
		if (sum_wealths_u < DELTA)
		{
			ugini = nodata_value;
		}
		else
		{
			float* xarry_u = new float[length];
			float* yarry_u = new float[length];
			float* yarry_u_max = new float[length];
			float* yarry_u_min = new float[length];
			int valid_value1 = length - 1;
			//get the upper and lower bounds of the integral for each step
			for (int ii = 0; ii < length; ii++)
			{
				xarry_u[ii] = (ii*1.0) / (valid_value1*1.0);
				yarry_u[ii] = cum_wealths_u[ii] / sum_wealths_u;
				if (xarry_u[ii] > yarry_u[ii])
				{
					yarry_u_max[ii] = xarry_u[ii];
					yarry_u_min[ii] = yarry_u[ii];
				}
				else
				{
					yarry_u_max[ii] = yarry_u[ii];
					yarry_u_min[ii] = xarry_u[ii];
				}
			}
			//calculate ugini
			ugini = 0;
			float u_max = get_trapz(yarry_u_max, xarry_u, length);
			float u_min = get_trapz(yarry_u_min, xarry_u, length);
			ugini = u_max - u_min;
			ugini *= 2.0;
			delete[]xarry_u;
			delete[]yarry_u;
			delete[]yarry_u_max;
			delete[]yarry_u_min;
		}
		delete[]cum_wealths_u;
		return ugini;
	}


	//Get the maximum and minimum values of unranked Gini index of every year
	float* stat_ugini(float* ts_daily, int nyears, int start_year, int ts_long, int nodata_value)
	{
		//get the extended sequence
		float* ts_daily_extend = extend_ts(ts_daily, nyears, start_year, ts_long);
		//the ugini_min is in the front, the ugini_max is in the back
		float* ugini_result = new float[2*nyears];
		//calculate maximum and minimum values of ugini index of every year
		int data_start0 = 0;
		for (int ii = 0; ii < nyears; ii++)
		{
			int days = get_days_year(ii + start_year);
			float* ugini = new float[days];
			int days1 = days + 1;
			float* prec_data = new float[days1];
			int data_start = data_start0;
			int data_end = data_start + days;
			int length = 0;
			//the first should be 0.0, which is used to calculate Gini
			prec_data[0] = 0.0;
			//get all data of the first time step
			for (int kk = data_start; kk < data_end; kk++)
			{
				if(ts_daily_extend[kk]<-1 || abs(ts_daily_extend[kk]-nodata_value)<DELTA){}
				else
				{
					length++;
					prec_data[length] = ts_daily_extend[kk];
				}
			}
			//calculate ugini index
			ugini[0] = get_ugini(prec_data, length+1, -9999);
			//get other ugini index of the year
			for (int jj = 1; jj < days; jj++)
			{
				//get the array and the length of the array of the time step
				length = add_remove_ugini(&(prec_data[1]), length, ts_daily_extend[data_start], ts_daily_extend[data_end], nodata_value);
				data_start++;
				data_end++;
				//calculate ugini index
				ugini[jj] = get_ugini(prec_data, length+1, -9999);
			}
			//the ugini_min is in the front, the ugini_max is in the back of the array
			ugini_result[ii] = get_array_min(ugini, days);
			ugini_result[ii+nyears] = get_array_max(ugini, days);
			data_start0 += days;
			delete[]ugini;
			delete[]prec_data;
		}
		delete[]ts_daily_extend;
		return ugini_result;
	}


	//Get the Gini index of the input daily-scale time series rainfall data
	float get_gini(float* ts_daily, int length, float nodata_value)
	{
		float gini = nodata_value;
		//calculate the cumulative value
		float* cum_wealths_u = new float[length];
		cumsum(ts_daily, cum_wealths_u, length);
		float sum_wealths_u = cum_wealths_u[length - 1];
		//calculate the Gini or wet-day Gini index
		if (sum_wealths_u < DELTA)
		{
			gini = nodata_value;
		}
		else
		{
			float* xarry = new float[length];
			float* yarry = new float[length];
			//calculate Gini or wet-day Gini index
			int valid_value1 = length - 1;
			for (int ii = 0; ii < length; ii++)
			{
				xarry[ii] = (ii*1.0) / (valid_value1*1.0);
				yarry[ii] = cum_wealths_u[ii] / sum_wealths_u;
			}
			gini = 0;
			float value_under = get_trapz(yarry, xarry, length);
			gini = 0.5 - value_under;
			gini *= 2.0;
			delete[]xarry;
			delete[]yarry;
		}
		delete[]cum_wealths_u;
		return gini;
	}


	//Get the maximum and minimum values of Gini index of every year
	float* stat_gini(float* ts_daily, int nyears, int start_year, int ts_long, int nodata_value)
	{
		//get the extended sequence
		float* ts_extend = extend_ts(ts_daily, nyears, start_year, ts_long);
		//the Gini_min is in the front, the Gini_max is in the back
		float* gini_result = new float[nyears*2];
		//calculate maximum and minimum values of Gini index of every year
		int data_start0 = 0;
		for (int ii = 0; ii < nyears; ii++)
		{
			int days = get_days_year(ii + start_year);
			float* gini = new float[days];
			int days1 = days + 1;
			float* prec_data = new float[days1];
			int data_start = data_start0;
			int data_end = data_start + days;
			//the first should be 0.0, which is used to calculate Gini
			prec_data[0] = 0.0;
			//get all data and the data number of the first time step
			int length = 0;
			for (int kk = data_start; kk < data_end; kk++)
			{
				if(ts_extend[kk]<-1 || abs(ts_extend[kk]-nodata_value)<DELTA){}
				else
				{
					length++;
					prec_data[length] = ts_extend[kk];
				}
				
			}
			//sort from smallest to largest
			sort(prec_data+1, prec_data+length+1);
			//calculate Gini index
			gini[0] = get_gini(prec_data, length+1, nodata_value);
			for (int jj = 1; jj < days; jj++)
			{
				//get the array and the length of the array of the time step
				length = add_remove(&(prec_data[1]), length, ts_extend[data_start], ts_extend[data_end], -1, nodata_value);
				data_start++;
				data_end++;
				//calculate Gini index
				gini[jj] = get_gini(prec_data, length+1, nodata_value);
			}
			//the Gini_min is in the front, the Gini_max is in the back of the array
			gini_result[ii] = get_array_min(gini, days);
			gini_result[ii+nyears] = get_array_max(gini, days);
			data_start0 += days;
			delete[]gini;
			delete[]prec_data;
		}
		delete[]ts_extend;
		return gini_result;
	}


	//Get the maximum and minimum values of wet-day Gini index of every year
	float* stat_wgini(float* ts_daily, int nyears, int start_year, int ts_long, int nodata_value, float threshold)
	{
		//get the extended sequence
		float* ts_extend = extend_ts(ts_daily, nyears, start_year, ts_long);
		//the wgini_min is in the front, the wgini_max is in the back
		float* wgini_result = new float[nyears*2];
		//calculate maximum and minimum values of Gini index of every year
		int data_start0 = 0;
		for (int ii = 0; ii < nyears; ii++)
		{
			int days = get_days_year(ii + start_year);
			float* wgini = new float[days];
			int days1 = days + 1;
			float* prec_data = new float[days1];
			int data_start = data_start0;
			int data_end = data_start + days;
			//the first should be 0.0, which is used to calculate Gini
			prec_data[0] = 0.0;
			//get all data and the data number of the first time step
			int length = 0;
			for(int kk = data_start; kk < data_end; kk++)
			{
				if(ts_extend[kk]<threshold || abs(ts_extend[kk]-nodata_value)<DELTA){}
				else
				{
					length++;
					prec_data[length] = ts_extend[kk];
				}
			}
			//sort from smallest to largest
			sort(prec_data+1, prec_data+length+1);
			//calculate wgini index
			wgini[0] = get_gini(prec_data, length+1, nodata_value);
			for (int jj = 1; jj < days; jj++)
			{
				//get the array and the length of the array of the time step
				length = add_remove(&(prec_data[1]), length, ts_extend[data_start], ts_extend[data_end], threshold, nodata_value);
				data_start++;
				data_end++;
				//calculate wgini index
				wgini[jj] = get_gini(prec_data, length+1, nodata_value);
			}
			wgini_result[ii] = get_array_min(wgini, days);
			wgini_result[ii+nyears] = get_array_max(wgini, days);
			data_start0 += days;
			delete[]wgini;
			delete[]prec_data;
		}
		delete[]ts_extend;
		return wgini_result;
	}
}