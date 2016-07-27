package com.zimmem;


import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Created by Zimmem on 2016/7/27.
 */
public class ThreadContext {

    private static ThreadLocal<Map<String, Object>> holder = new ThreadLocal<>();


    public static void set(String name, Object o){
        Optional.ofNullable(holder.get()).orElseGet(() ->{
            Map<String, Object> map = new HashMap<>();
            holder.set(map);
            return map;
        }).put(name, o);
    }

    public static  Object get(String name){
        return Optional.ofNullable(holder.get()).orElse((Map<String, Object>) Collections.EMPTY_MAP).get(name);
    }

    public static void clear(){
        holder.remove();
    }

}
